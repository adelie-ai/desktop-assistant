//! The reads and writes behind the `[Recall]` block's skill arm (#1154).
//!
//! `PgSkillIndexStore::nearest_by_embedding` is what a turn asks of the skill
//! catalog before the model's first move, and `PgSkillUseLog` is what records
//! the offer it makes and the open that may follow. This suite pins the
//! properties the arm's correctness rests on, and the ones no unit test can
//! reach because they live in SQL.
//!
//! 1. **Only an approved skill takes part.** Approval is consent, and it is a
//!    separate axis from provenance: a skill Adele wrote for herself is
//!    `TrustTier::Local`, the most trusted provenance there is, and must not be
//!    offered until a person says so. The scan excludes an unapproved row from
//!    the candidates *and* from the spread they are graded against.
//! 2. **One user's skills and no other's.** The catalog is host-global with a
//!    per-row owner, so the scope predicate is the only guard.
//! 3. **One row per name.** The catalog can hold a global skill and a user's own
//!    under one name, and `builtin_skill_get` resolves that to the user's own.
//!    Two lines for one openable procedure would be two lines the model cannot
//!    tell apart.
//! 4. **Nearest first, with a usable distance and the catalog's own spread.**
//!    The bar reads each distance against the source's own dispersion, so an
//!    unordered result or a spread measured over another source would make it
//!    meaningless.
//! 5. **An open counts only against a standing offer**, and counting it takes
//!    the offer down - so a retried read is one open, and a read the block
//!    never offered is not an open at all.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test skill_recall
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::{
    IndexedSkill, Locality, SkillApproval, SkillKind, SkillScope, TrustTier,
};
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_core::ports::knowledge_use::OfferScope;
use desktop_assistant_core::ports::recall::RECALL_DISPERSION_MIN_ROWS;
use desktop_assistant_core::ports::skill_index::SkillIndexStore;
use desktop_assistant_core::ports::skill_use::SkillUseLog;
use desktop_assistant_storage::{PgSkillIndexStore, PgSkillUseLog};
use pgvector::Vector;

/// A synthetic tenant, never a real identity.
const USER: &str = "skill-recall-user";
const OTHER_USER: &str = "skill-recall-other-user";
const CONVERSATION: &str = "conv-1";

/// The model every seeded vector is stamped with, and the one every read passes.
const MODEL: &str = "skill-recall-test-model";

async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("skillrecall1154").await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set");
    }
    fx
}

/// A three-dimensional unit vector pointing along one axis. Cosine distance
/// between two of these is 1.0; between a vector and itself it is 0.0.
fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 3];
    v[i] = 1.0;
    v
}

/// A three-dimensional unit vector `radians` around from [`axis`] zero, so a
/// fixture can seed a spread of distances rather than only 0.0 and 1.0.
fn at_angle(radians: f32) -> Vec<f32> {
    vec![radians.cos(), radians.sin(), 0.0]
}

/// A catalog row, unapproved as every write path leaves one, and locally
/// authored as a skill somebody wrote on this machine is.
fn a_skill(name: &str, description: &str, owner: Option<&str>) -> IndexedSkill {
    IndexedSkill {
        name: name.to_string(),
        description: description.to_string(),
        kind: SkillKind::Skill,
        disk_path: format!("/usr/share/adelie/skills/{name}/SKILL.md"),
        owner_user_id: owner.map(str::to_string),
        locality: Locality::Daemon,
        content_hash: format!("hash-{name}"),
        // The most trusted provenance there is, on every fixture, so no test
        // below can pass by accident on a tier check standing in for approval.
        trust_tier: TrustTier::Local,
        source: Some("system".to_string()),
        tags: vec!["ops".to_string()],
        attachments: vec![],
        body: format!("# {name}\n\nA long body nothing in the recall arm may read.\n"),
        metadata: serde_json::json!({}),
        present_on_disk: true,
        last_seen_at: None,
        approved_at: None,
        approved_by: None,
    }
}

fn now() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339("2026-08-07T12:00:00Z")
        .expect("a fixed clock parses")
        .with_timezone(&chrono::Utc)
}

/// Write a skill, stamp it with an embedding, and approve it unless
/// `approved` says otherwise.
async fn seed(
    store: &PgSkillIndexStore,
    pool: &sqlx::PgPool,
    skill: &IndexedSkill,
    embedding: Vec<f32>,
    approved: bool,
) {
    store.upsert(skill, now()).await.expect("seed the catalog");
    sqlx::query(
        "UPDATE skill_index SET embedding = ARRAY[$1]::vector[], embedding_model = $2 \
         WHERE name = $3 AND owner_key = COALESCE($4, '')",
    )
    .bind(Vector::from(embedding))
    .bind(MODEL)
    .bind(&skill.name)
    .bind(skill.owner_user_id.as_deref())
    .execute(pool)
    .await
    .expect("stamp the embedding");
    if approved {
        store
            .set_approval(
                &SkillScope::of(skill),
                std::slice::from_ref(&skill.name),
                Some(SkillApproval {
                    at: now(),
                    by: Some("a-person".to_string()),
                }),
            )
            .await
            .expect("approve the skill");
    }
}

// --- The scan: what may be offered, and what a distance is worth ------------

/// Acceptance (#1154): a skill nobody has approved never reaches the block.
///
/// Approval is the axis that says a person agreed the procedure may be
/// followed. `builtin_skill_get` refuses an unapproved skill's body, so a line
/// offering it is a line the model can only fail on - and it would accrue an
/// offer every turn it ranked near a prompt and never an open, which is the
/// profile ranking reads as evidence to retire an entry.
#[tokio::test]
async fn an_unapproved_skill_is_absent_from_the_recall_scan() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("blessed", "A procedure a person approved.", None),
            axis(0),
            true,
        )
        .await;
        seed(
            &store,
            &fx.pool,
            &a_skill("pending", "A procedure nobody approved.", None),
            axis(0),
            false,
        )
        .await;

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");

        let names: Vec<&str> = found.skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["blessed"], "only the approved skill is offered");
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance (#1154): approval is a distinct axis from `trust_tier`, and a
/// locally authored skill is not approved by virtue of being local.
///
/// Both fixtures below are `TrustTier::Local`, which is the most trusted
/// provenance the catalog has. The one nobody approved is still withheld, so
/// nothing here can be passing on a tier check.
#[tokio::test]
async fn a_locally_authored_skill_is_not_offered_by_virtue_of_being_local() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        let authored = a_skill("self-authored", "A procedure Adele wrote.", None);
        assert_eq!(
            authored.trust_tier,
            TrustTier::Local,
            "precondition: the most trusted provenance there is"
        );
        // `write_authored` is the path `promote_plan_to_skill` uses, and it
        // forces approval clear whatever the caller passes.
        store
            .write_authored(&authored, now())
            .await
            .expect("author the skill");
        sqlx::query(
            "UPDATE skill_index SET embedding = ARRAY[$1]::vector[], embedding_model = $2 \
             WHERE name = 'self-authored'",
        )
        .bind(Vector::from(axis(0)))
        .bind(MODEL)
        .execute(&fx.pool)
        .await
        .expect("stamp the embedding");

        let stored = store
            .get("self-authored", None)
            .await
            .expect("read it back")
            .expect("the row exists");
        assert_eq!(stored.trust_tier, TrustTier::Local);
        assert!(
            !stored.is_approved(),
            "provenance is not consent: a self-authored skill is born unapproved"
        );

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");
        assert!(
            found.skills.is_empty(),
            "a locally authored skill nobody approved is not offered"
        );
    })
    .await;

    fx.cleanup().await;
}

/// The scan reads the calling user's own skills and the global ones, and no
/// other user's.
#[tokio::test]
async fn the_recall_scan_reads_no_other_users_skills() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    with_user_id(UserId::new(OTHER_USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("theirs", "Another tenant's procedure.", Some(OTHER_USER)),
            axis(0),
            true,
        )
        .await;
    })
    .await;
    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("mine", "My own procedure.", Some(USER)),
            axis(0),
            true,
        )
        .await;
        seed(
            &store,
            &fx.pool,
            &a_skill("everyones", "A host-global procedure.", None),
            axis(0),
            true,
        )
        .await;

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");

        let mut names: Vec<&str> = found.skills.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["everyones", "mine"]);
    })
    .await;

    fx.cleanup().await;
}

/// One name, one line. A user's own skill shadows a global one of the same
/// name, because that is what `builtin_skill_get` resolves to - so the line the
/// block shows is the procedure a fetch would return.
#[tokio::test]
async fn the_recall_scan_answers_one_row_per_name_preferring_the_users_own() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("deploy", "The host-global copy.", None),
            axis(0),
            true,
        )
        .await;
        seed(
            &store,
            &fx.pool,
            &a_skill("deploy", "My own copy.", Some(USER)),
            axis(0),
            true,
        )
        .await;

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");

        assert_eq!(found.skills.len(), 1, "one name is one line");
        assert_eq!(found.skills[0].description, "My own copy.");
    })
    .await;

    fx.cleanup().await;
}

/// Nearest first, with the catalog's own spread beside the candidates. The bar
/// is dimensionless, so a distance means nothing until it is read against the
/// source it came from.
#[tokio::test]
async fn the_recall_scan_answers_nearest_first_with_the_catalogs_own_spread() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        // Enough rows, spread in angle, for a median absolute deviation over
        // them to be a measurement rather than noise.
        let rows = RECALL_DISPERSION_MIN_ROWS + 5;
        for i in 0..rows {
            let radians = 0.05 + (i as f32) * 0.03;
            seed(
                &store,
                &fx.pool,
                &a_skill(
                    &format!("procedure-{i:02}"),
                    "A procedure in a catalog of procedures.",
                    None,
                ),
                at_angle(radians),
                true,
            )
            .await;
        }

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");

        assert_eq!(found.skills.len(), 10, "the limit bounds what comes back");
        let distances: Vec<f64> = found
            .skills
            .iter()
            .map(|s| s.distance.expect("a measured candidate carries a distance"))
            .collect();
        assert!(
            distances.windows(2).all(|w| w[0] <= w[1]),
            "nearest first, which is what the bar rests on: {distances:?}"
        );
        let dispersion = found
            .dispersion
            .expect("a catalog of this size states its own geometry");
        // Read back through the quantity the bar compares: the nearest row
        // stands further out of this catalog than the tenth does.
        assert!(
            dispersion.deviations_below_median(distances[0])
                > dispersion.deviations_below_median(distances[9]),
            "the spread grades the candidates it travelled with"
        );
    })
    .await;

    fx.cleanup().await;
}

/// The degraded read - what runs when no embedding is available - applies the
/// same approval rule. A backend outage must not turn into an offer of
/// something nobody approved.
#[tokio::test]
async fn the_degraded_full_text_read_also_withholds_an_unapproved_skill() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("blessed", "How to publish a crate to the registry.", None),
            axis(0),
            true,
        )
        .await;
        seed(
            &store,
            &fx.pool,
            &a_skill("pending", "How to publish a crate to the registry.", None),
            axis(0),
            false,
        )
        .await;

        let found = store
            .search_text_any_term("how do I publish a crate?", 10)
            .await
            .expect("the degraded read answers");

        let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["blessed"]);
        assert!(
            found[0].distance.is_none(),
            "a lexical match carries no distance to read against a spread"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance (#1154): a skill that came from outside this machine is not
/// offered.
///
/// The block is a system message with no tool call in it, so nothing taints
/// and the tool gate stays open. `builtin_skill_search` returns the same
/// `description` field and is classified `Declared(SkillTrustTier)`, which
/// means the platform already rules that a non-local skill's text is
/// third-party content that must close the gate. Delivering the same bytes
/// through a more trusted channel with less checking is the defect this
/// excludes.
#[tokio::test]
async fn a_skill_from_outside_this_machine_is_absent_from_the_recall_scan() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("written-here", "A procedure somebody wrote here.", None),
            axis(0),
            true,
        )
        .await;
        let mut installed = a_skill(
            "from-github",
            "A procedure fetched from a repository.",
            None,
        );
        installed.trust_tier = TrustTier::Github;
        seed(&store, &fx.pool, &installed, axis(0), true).await;

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");
        let names: Vec<&str> = found.skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["written-here"]);

        let lexical = store
            .search_text_any_term("a procedure", 10)
            .await
            .expect("the degraded read answers");
        let names: Vec<&str> = lexical.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["written-here"],
            "a backend outage must not turn into an offer of third-party text"
        );
    })
    .await;

    fx.cleanup().await;
}

/// A non-local skill shadowing a local one by name must not leave the block
/// offering the local line while a fetch hands back the non-local body.
///
/// The trust rule therefore applies to the row a name resolved to, not to the
/// set it resolves from: the name is dropped outright rather than falling
/// through to the row underneath it.
#[tokio::test]
async fn a_name_whose_resolved_row_came_from_outside_is_dropped_rather_than_falling_through() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("deploy", "The host-global copy, written here.", None),
            axis(0),
            true,
        )
        .await;
        let mut mine = a_skill(
            "deploy",
            "My own copy, installed from elsewhere.",
            Some(USER),
        );
        mine.trust_tier = TrustTier::Github;
        seed(&store, &fx.pool, &mine, axis(0), true).await;

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");

        assert!(
            found.skills.is_empty(),
            "the fetch would return the installed personal row, so offering the global \
             line would describe a procedure the model will not be given: {:?}",
            found.skills
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance (#1154): the line the block offers describes the procedure a
/// fetch hands back.
///
/// `builtin_skill_get` prefers the caller's own row only while it is usable -
/// on disk and approved - and falls back to the global one otherwise. The scan
/// resolves a duplicated name by that same rule. Three states where a naive
/// "the personal row always wins" would disagree.
#[tokio::test]
async fn the_scan_resolves_a_duplicated_name_the_way_a_fetch_does() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());

    // A personal tombstone must not shadow a live global skill.
    with_user_id(UserId::new(USER), async {
        let mut mine = a_skill("deploy", "My own copy, files gone.", Some(USER));
        mine.present_on_disk = false;
        seed(&store, &fx.pool, &mine, axis(0), true).await;
        // `upsert` marks a row present, so the tombstone is set afterwards the
        // way a reconcile pass sets it.
        store
            .set_presence(
                &SkillScope::Owner(USER.to_string()),
                &["deploy".to_string()],
                false,
            )
            .await
            .expect("mark the personal row absent");
        seed(
            &store,
            &fx.pool,
            &a_skill("deploy", "The host-global copy, live.", None),
            axis(0),
            true,
        )
        .await;

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");
        assert_eq!(found.skills.len(), 1);
        assert_eq!(
            found.skills[0].description, "The host-global copy, live.",
            "the fetch reaches past a dead personal row, so the line must too"
        );
        assert!(
            found.skills[0].present_on_disk,
            "and the line must not wear the tombstone's marker"
        );
    })
    .await;

    // An unapproved personal row must not shadow a live global skill either.
    with_user_id(UserId::new(OTHER_USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("rotate", "My own draft, unapproved.", Some(OTHER_USER)),
            axis(0),
            false,
        )
        .await;
        seed(
            &store,
            &fx.pool,
            &a_skill("rotate", "The host-global copy, approved.", None),
            axis(0),
            true,
        )
        .await;

        let found = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the scan answers");
        let rotate: Vec<&str> = found
            .skills
            .iter()
            .filter(|s| s.name == "rotate")
            .map(|s| s.description.as_str())
            .collect();
        assert_eq!(rotate, vec!["The host-global copy, approved."]);
    })
    .await;

    fx.cleanup().await;
}

// --- The use log: offers and opens ------------------------------------------

/// Acceptance (#1154): offers and opens for skills reach the use log, and a
/// skill opened by id after being offered records an open.
#[tokio::test]
async fn a_skill_offered_by_the_block_and_then_opened_records_an_open() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());
    let log = PgSkillUseLog::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("deploy", "How to deploy.", None),
            axis(0),
            true,
        )
        .await;

        let offered = log
            .record_offered(OfferScope::recall(CONVERSATION), vec!["deploy".to_string()])
            .await
            .expect("the offer is recorded");
        assert_eq!(offered, 1);

        let opened = log
            .record_opened(CONVERSATION.to_string(), vec!["deploy".to_string()])
            .await
            .expect("the open is recorded");
        assert_eq!(opened, 1);

        let records = log
            .records(vec!["deploy".to_string()])
            .await
            .expect("the log answers");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry_id, "deploy");
        assert_eq!(records[0].offered_count, 1);
        assert_eq!(records[0].opened_count, 1);
        assert_eq!(
            records[0].recent_uses.len(),
            1,
            "the open lands in the window the activation score reads"
        );
    })
    .await;

    fx.cleanup().await;
}

/// A skill surfaced and never opened is visible as such: the offer counter
/// moves and the open counter does not.
#[tokio::test]
async fn a_skill_offered_and_never_opened_records_the_offer_alone() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());
    let log = PgSkillUseLog::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("ignored", "A procedure nobody reads.", None),
            axis(0),
            true,
        )
        .await;

        for _ in 0..3 {
            log.record_offered(
                OfferScope::recall(CONVERSATION),
                vec!["ignored".to_string()],
            )
            .await
            .expect("the offer is recorded");
        }

        let records = log
            .records(vec!["ignored".to_string()])
            .await
            .expect("the log answers");
        assert_eq!(records[0].offered_count, 3);
        assert_eq!(records[0].opened_count, 0);
        assert!(
            records[0].recent_uses.is_empty(),
            "an offer nobody took up is not a use"
        );
    })
    .await;

    fx.cleanup().await;
}

/// A read nothing offered is not an open. The model opens a skill for many
/// reasons - a search hit, a name it already held - and only a taken-up offer
/// is evidence that the block worked.
#[tokio::test]
async fn a_skill_read_with_no_standing_offer_records_no_open() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());
    let log = PgSkillUseLog::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("deploy", "How to deploy.", None),
            axis(0),
            true,
        )
        .await;

        let opened = log
            .record_opened(CONVERSATION.to_string(), vec!["deploy".to_string()])
            .await
            .expect("the write succeeds");

        assert_eq!(opened, 0, "no offer stood, so nothing was taken up");
        assert!(
            log.records(vec!["deploy".to_string()])
                .await
                .expect("the log answers")
                .is_empty(),
            "an unoffered read leaves no record at all"
        );
    })
    .await;

    fx.cleanup().await;
}

/// A second read of the same skill in the same turn is one open: counting an
/// open takes the offer down, so a retried tool call adds nothing.
#[tokio::test]
async fn a_second_read_of_one_offered_skill_records_one_open() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());
    let log = PgSkillUseLog::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("deploy", "How to deploy.", None),
            axis(0),
            true,
        )
        .await;
        log.record_offered(OfferScope::recall(CONVERSATION), vec!["deploy".to_string()])
            .await
            .expect("the offer is recorded");

        for _ in 0..2 {
            log.record_opened(CONVERSATION.to_string(), vec!["deploy".to_string()])
                .await
                .expect("the write succeeds");
        }

        let records = log
            .records(vec!["deploy".to_string()])
            .await
            .expect("the log answers");
        assert_eq!(records[0].opened_count, 1);
    })
    .await;

    fx.cleanup().await;
}

/// A recall offer replaces the conversation's standing skill offers, so an open
/// can only ever be credited to the turn that made the offer.
#[tokio::test]
async fn a_recall_offer_replaces_the_conversations_standing_skill_offers() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());
    let log = PgSkillUseLog::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        for name in ["first-turn", "second-turn"] {
            seed(
                &store,
                &fx.pool,
                &a_skill(name, "A procedure.", None),
                axis(0),
                true,
            )
            .await;
        }

        log.record_offered(
            OfferScope::recall(CONVERSATION),
            vec!["first-turn".to_string()],
        )
        .await
        .expect("the first turn's offer");
        log.record_offered(
            OfferScope::recall(CONVERSATION),
            vec!["second-turn".to_string()],
        )
        .await
        .expect("the second turn's offer");

        let stale = log
            .record_opened(CONVERSATION.to_string(), vec!["first-turn".to_string()])
            .await
            .expect("the write succeeds");
        assert_eq!(stale, 0, "the previous turn's offer no longer stands");

        let live = log
            .record_opened(CONVERSATION.to_string(), vec!["second-turn".to_string()])
            .await
            .expect("the write succeeds");
        assert_eq!(live, 1, "this turn's offer is the one that can be taken up");
    })
    .await;

    fx.cleanup().await;
}

/// A name the catalog does not hold in the caller's scope records nothing. The
/// catalog is host-global, so a name is not evidence that the caller may see
/// the skill.
#[tokio::test]
async fn an_offer_of_another_users_skill_records_nothing() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());
    let log = PgSkillUseLog::new(fx.pool.clone());

    with_user_id(UserId::new(OTHER_USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("theirs", "Another tenant's procedure.", Some(OTHER_USER)),
            axis(0),
            true,
        )
        .await;
    })
    .await;

    with_user_id(UserId::new(USER), async {
        let offered = log
            .record_offered(OfferScope::recall(CONVERSATION), vec!["theirs".to_string()])
            .await
            .expect("the write succeeds");

        assert_eq!(offered, 0);
        assert!(
            log.records(vec!["theirs".to_string()])
                .await
                .expect("the log answers")
                .is_empty()
        );
    })
    .await;

    fx.cleanup().await;
}

/// The log is per-user: one person's opens say nothing about another's.
#[tokio::test]
async fn the_skill_use_log_reads_only_the_calling_users_own_record() {
    let Some(fx) = fixture().await else { return };
    let store = PgSkillIndexStore::new(fx.pool.clone());
    let log = PgSkillUseLog::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        seed(
            &store,
            &fx.pool,
            &a_skill("everyones", "A host-global procedure.", None),
            axis(0),
            true,
        )
        .await;
        log.record_offered(
            OfferScope::recall(CONVERSATION),
            vec!["everyones".to_string()],
        )
        .await
        .expect("the offer is recorded");
    })
    .await;

    with_user_id(UserId::new(OTHER_USER), async {
        assert!(
            log.records(vec!["everyones".to_string()])
                .await
                .expect("the log answers")
                .is_empty(),
            "another person's offer is not this person's history"
        );
    })
    .await;

    fx.cleanup().await;
}
