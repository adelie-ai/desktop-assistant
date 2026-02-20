import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import QtCore
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore
import org.kde.plasma.components as PlasmaComponents
import org.kde.plasma.plasma5support as Plasma5Support

Item {
    id: root
    clip: true
    Kirigami.Theme.colorSet: Kirigami.Theme.View
    Kirigami.Theme.inherit: false
    property bool panelMode: false
    implicitWidth: 520
    implicitHeight: 620
    readonly property color themeBackgroundColor: Kirigami.Theme.backgroundColor
    readonly property color themeTextColor: Kirigami.Theme.textColor
    readonly property color themeDisabledTextColor: Kirigami.Theme.disabledTextColor
    readonly property color themeHighlightColor: Kirigami.Theme.highlightColor
    readonly property color themeHighlightedTextColor: Kirigami.Theme.highlightedTextColor

    property string helperPath: Qt.resolvedUrl("../code/dbus_client.py").toString().replace("file://", "")
    property string productionService: "org.desktopAssistant"
    property string developmentService: "org.desktopAssistant.Dev"
    property string activeService: productionService
    property bool serviceInitialized: false
    property bool productionServiceRunning: false
    property bool devServiceRunning: false
    readonly property bool hideWidget: false
    property var serviceChoices: []
    property string conversationId: ""
    property bool busy: false
    property bool loadingConversation: false
    property double loadingConversationStartedAtMs: 0
    property int conversationLoadSequence: 0
    property bool initialLoadAutoScrollPending: true
    property bool debugEnabled: false
    property bool serviceStatusRequestInFlight: false
    property int currentMessageCount: 0
    property var transcriptEntries: []
    property int transcriptEntryIdSeq: 0
    property var expandedToolEntries: ({})
    // NOTE: must NOT be readonly — QV4 does not reliably root readonly var properties
    // under GC pressure, leading to SEGV in Object::insertMember when writing to
    // pending[cmd] during a GC cycle triggered by heavy Object.assign usage.
    property var pending: ({})
    readonly property int maxTranscriptEntries: 400
    // Configurable UI back-load limit. 0 means full history.
    // Default is 50 to balance responsiveness and context visibility.
    readonly property int defaultMaxRenderedMessages: 50
    readonly property int maxLoadedMessageChars: 12000
    readonly property int maxLiveMessageChars: 12000
    readonly property int maxDebugPayloadChars: 1200
    readonly property int conversationLoadTimeoutMs: 15000
    property string promptText: ""
    property var conversationChoices: []
    property real uiScale: 1.0
    readonly property string conversationTitle: panelMode ? "Panel Chat" : "Desktop Chat"
    readonly property real awaitTimeoutSeconds: panelMode ? 45.0 : 60.0
    readonly property real minUiScale: 0.9
    readonly property real maxUiScale: 1.35
    readonly property real zoomStep: 0.05
    readonly property string adeleAvatarSource: Qt.resolvedUrl("../images/adele.png")
    property string configuredUserAvatarPath: String(Plasmoid.configuration.userAvatarPath || "").trim()
    readonly property real baseFontPointSize: Math.max(1, Number(Kirigami.Theme.defaultFont.pointSize || Qt.application.font.pointSize || 10))
    readonly property int scaledTopIconSize: Math.max(16, Math.round(24 * uiScale))
    readonly property int scaledHeaderIconSize: Math.max(64, Math.round(96 * uiScale))
    readonly property bool ultraNarrow: width > 0 && width < 430
    readonly property real transcriptAvatarSize: 24 * uiScale
    readonly property real transcriptBubbleSpacing: 6
    readonly property real transcriptWideBubbleWidth: Math.max(120, transcript.width)
    readonly property real transcriptMessageBubbleWidth: {
        const available = Math.max(120, transcript.width - (transcriptAvatarSize + transcriptBubbleSpacing))
        return Math.max(120, available * 0.88)
    }
    readonly property string homeDirectory: StandardPaths.writableLocation(StandardPaths.HomeLocation)
    readonly property string accountName: {
        const trimmedHome = String(homeDirectory || "").replace(/\/+$/, "")
        const chunks = trimmedHome.split("/").filter(function(chunk) {
            return chunk.length > 0
        })
        return chunks.length > 0 ? chunks[chunks.length - 1] : ""
    }
    readonly property bool hasRealMessages: {
        for (let i = 0; i < transcriptEntries.length; i++) {
            if (transcriptEntries[i].kind === "message") return true
        }
        return false
    }
    property int lateResponsePollRemaining: 0

    function toImageSource(pathValue) {
        const value = String(pathValue || "").trim()
        if (value.length === 0) {
            return ""
        }
        if (value.indexOf("file://") === 0 || value.indexOf("image://") === 0 || value.indexOf("qrc:/") === 0 || value.indexOf(":/") === 0) {
            return value
        }
        if (value[0] === "/") {
            return "file://" + value
        }
        return value
    }

    function userAvatarCandidates() {
        const candidates = []
        const configured = toImageSource(configuredUserAvatarPath)
        if (configured.length > 0) {
            candidates.push(configured)
        }
        if (accountName.length > 0) {
            candidates.push(toImageSource("/var/lib/AccountsService/icons/" + accountName))
        }
        candidates.push(toImageSource(homeDirectory + "/.face.icon"))
        candidates.push(toImageSource(homeDirectory + "/.face"))
        return candidates
    }

    function shellEscape(value) {
        return "'" + value.replace(/'/g, "'\\''") + "'"
    }

    function limitDebugPayload(text) {
        const normalized = String(text || "")
        if (normalized.length <= maxDebugPayloadChars) {
            return normalized
        }
        return normalized.substring(0, maxDebugPayloadChars) + "…"
    }

    function parseToolCommand(command) {
        const normalized = String(command || "").trim()
        const helperMatch = normalized.match(/^python3\s+'([^']+)'\s+--service\s+'([^']+)'\s*(.*)$/)
        if (helperMatch) {
            const args = String(helperMatch[3] || "").trim()
            return {
                toolName: "dbus_client.py",
                inputText: args.length > 0
                    ? ("service=" + helperMatch[2] + "\nargs: " + args)
                    : ("service=" + helperMatch[2]),
            }
        }

        const settingsMatch = normalized.match(/^systemsettings\s+(.+)$/)
        if (settingsMatch) {
            return {
                toolName: "systemsettings",
                inputText: settingsMatch[1],
            }
        }

        return {
            toolName: "shell",
            inputText: normalized,
        }
    }

    function appendToolExecutionDebug(phase, command, details) {
        if (!debugEnabled) {
            return
        }
        const parsed = parseToolCommand(command)
        const lines = [
            "tool: " + parsed.toolName,
            "phase: " + phase,
            "input:",
            limitDebugPayload(parsed.inputText),
        ]
        const trimmedDetails = String(details || "").trim()
        if (trimmedDetails.length > 0) {
            lines.push("result:")
            lines.push(limitDebugPayload(trimmedDetails))
        }
        appendMessage("tool", lines.join("\n"), {
            toolName: parsed.toolName,
        })
    }

    function runCommand(command, onSuccess, onError, logDebug) {
        const shouldLogDebug = logDebug !== false
        if (shouldLogDebug) {
            appendToolExecutionDebug("run", command, "")
        }
        pending[command] = {
            success: onSuccess,
            error: onError,
            debug: shouldLogDebug,
        }
        executable.connectSource(command)
    }

    function helperCommand(commandText) {
        return "python3 "
            + shellEscape(helperPath)
            + " --service "
            + shellEscape(activeService)
            + " "
            + commandText
    }

    function zoomInUi() {
        uiScale = Math.min(maxUiScale, uiScale + zoomStep)
    }

    function zoomOutUi() {
        uiScale = Math.max(minUiScale, uiScale - zoomStep)
    }

    function resetZoomUi() {
        uiScale = 1.0
    }

    function maxSessionAgeDays() {
        const configured = Number(Plasmoid.configuration.maxSessionAgeDays)
        if (!Number.isFinite(configured) || configured < 0) {
            return 7
        }
        return Math.floor(configured)
    }

    function maxRenderedMessages() {
        const configured = Number(Plasmoid.configuration.maxRenderedMessages)
        if (!Number.isFinite(configured) || configured < 0) {
            return defaultMaxRenderedMessages
        }
        return Math.floor(configured)
    }

    function markdownListLineCount(textValue) {
        const normalized = String(textValue === undefined || textValue === null ? "" : textValue)
            .replace(/\r\n/g, "\n")
            .replace(/\r/g, "\n")
        if (normalized.length === 0) {
            return 0
        }

        const lines = normalized.split("\n")
        let count = 0
        for (let i = 0; i < lines.length; i++) {
            if (/^\s{0,3}(?:[-*+]|\d+[.)])\s+/.test(lines[i])) {
                count = count + 1
            }
        }
        return count
    }

    function shouldRenderAssistantAsMarkdown(textValue) {
        const normalized = String(textValue === undefined || textValue === null ? "" : textValue)
        if (normalized.length === 0) {
            return false
        }

        const listLines = markdownListLineCount(normalized)
        const hasLargeList = listLines >= 45 && normalized.length >= 1800
        return !hasLargeList
    }

    function loadPersistedService() {
        const persisted = String(Plasmoid.configuration.selectedService || "").trim()
        if (persisted === productionService || persisted === developmentService) {
            activeService = persisted
        }
        serviceInitialized = true
    }

    function persistActiveService() {
        if (Plasmoid.configuration.selectedService !== activeService) {
            Plasmoid.configuration.selectedService = activeService
        }
    }

    function serviceIndexByValue(serviceName) {
        for (let i = 0; i < serviceChoices.length; i++) {
            if (serviceChoices[i].value === serviceName) {
                return i
            }
        }
        return -1
    }

    function sameServiceChoices(left, right) {
        if (!left || !right || left.length !== right.length) {
            return false
        }
        for (let i = 0; i < left.length; i++) {
            if (left[i].value !== right[i].value || left[i].label !== right[i].label) {
                return false
            }
        }
        return true
    }

    function syncServicePicker() {
        const idx = serviceIndexByValue(activeService)
        if (idx >= 0) {
            servicePicker.currentIndex = idx
        }
    }

    function switchService(index) {
        if (index < 0 || index >= serviceChoices.length || busy) {
            return
        }

        const selectedService = serviceChoices[index].value
        if (selectedService === developmentService && !devServiceRunning) {
            appendStatus("Development service is not running")
            syncServicePicker()
            return
        }
        if (selectedService === activeService) {
            return
        }

        activeService = selectedService
        persistActiveService()
        conversationId = ""
        promptText = ""
        currentMessageCount = 0
        transcriptEntries = [
            {
                kind: "status",
                role: "status",
                text: "Switched to " + activeService,
            }
        ]
        reloadConversationList()
        refreshConversation()
    }

    function refreshServiceStatus(onReady, options) {
        const opts = options || {}
        const silent = opts.silent === true
        if (serviceStatusRequestInFlight) {
            if (onReady) {
                onReady()
            }
            return
        }
        serviceStatusRequestInFlight = true

        const command = "python3 " + shellEscape(helperPath) + " status"
        runCommand(
            command,
            function(stdout) {
                try {
                    const payload = JSON.parse(stdout)
                    productionServiceRunning = !!payload.production_running
                    devServiceRunning = !!payload.dev_running
                    if (payload.default_service && payload.default_service.length > 0) {
                        productionService = payload.default_service
                    }
                    if (payload.dev_service && payload.dev_service.length > 0) {
                        developmentService = payload.dev_service
                    }

                    let nextServiceChoices = [
                        {
                            value: productionService,
                            label: "Production",
                        }
                    ]

                    if (devServiceRunning) {
                        nextServiceChoices = nextServiceChoices.concat([
                            {
                                value: developmentService,
                                label: "Development",
                            }
                        ])
                    }

                    if (!sameServiceChoices(serviceChoices, nextServiceChoices)) {
                        serviceChoices = nextServiceChoices
                    }

                    if (!serviceInitialized) {
                        loadPersistedService()
                    }

                    if (activeService !== productionService && activeService !== developmentService) {
                        activeService = String(payload.selected_service || productionService)
                    }

                    if (!devServiceRunning && activeService === developmentService) {
                        if (!silent) {
                            appendStatus("Development service stopped; switching to production")
                        }
                        activeService = productionService
                        persistActiveService()
                    }

                    syncServicePicker()
                } catch (parseError) {
                    if (!silent) {
                        appendStatus("Service status parse error: " + parseError)
                    }
                } finally {
                    serviceStatusRequestInFlight = false
                    if (onReady) {
                        onReady()
                    }
                }
            },
            function(stderr) {
                serviceStatusRequestInFlight = false
                if (!silent) {
                    appendStatus("Service status error: " + stderr)
                }
                if (onReady) {
                    onReady()
                }
            },
            false
        )
    }

    function openSettingsDialog() {
        runCommand(
            "systemsettings kcm_desktopassistant",
            function(_stdout) {},
            function(_stderr) {
                appendStatus("Settings dialog failed to open")
            }
        )
    }

    function appendMessage(role, text, meta) {
        const entry = buildMessageEntry(role, text, meta)
        if (!entry) {
            return
        }
        appendTranscriptEntry(entry)
    }

    function buildMessageEntry(role, text, meta) {
        const normalizedText = String(text === undefined || text === null ? "" : text)
        const clippedText = normalizedText.length > maxLiveMessageChars
            ? normalizedText.substring(0, maxLiveMessageChars) + "\n\n[…message truncated for widget stability…]"
            : normalizedText
        if (clippedText.trim().length === 0) {
            return null
        }
        // NOTE: build in a single Object.assign expression — do NOT create a temp
        // and then mutate it in a second step; QV4 does not reliably root JS stack
        // locals under GC pressure, which causes SEGV in Object::insertMember.
        return Object.assign({ kind: "message", role: role, text: clippedText }, meta || {})
    }

    function appendStatus(text) {
        appendTranscriptEntry({
            kind: "status",
            role: "status",
            text: text,
        })
    }

    function clipLoadedMessageText(textValue) {
        const normalized = String(textValue === undefined || textValue === null ? "" : textValue)
        if (normalized.length <= maxLoadedMessageChars) {
            return normalized
        }
        return normalized.substring(0, maxLoadedMessageChars) + "\n\n[…message truncated for performance…]"
    }

    function transcriptIsAtBottom() {
        if (!transcript) {
            return true
        }
        const viewportHeight = Math.max(0, Number(transcript.height || 0))
        const contentHeight = Math.max(0, Number(transcript.contentHeight || 0))
        const maxContentY = Math.max(0, contentHeight - viewportHeight)
        const currentY = Math.max(0, Number(transcript.contentY || 0))
        return maxContentY <= 0 || currentY >= maxContentY - 2 || transcript.atYEnd
    }

    function appendTranscriptEntries(entries, stickIfAlreadyAtBottom) {
        if (!entries || entries.length === 0) {
            return
        }
        const previousContentY = Math.max(0, Number(transcript ? transcript.contentY : 0))
        const shouldStickToBottom = stickIfAlreadyAtBottom === true && transcriptIsAtBottom()
        const preparedEntries = []
        for (let i = 0; i < entries.length; i++) {
            if (!entries[i]) {
                continue
            }
            // Compute the id before Object.assign so the whole object is built
            // in one expression — see NOTE in buildMessageEntry about GC safety.
            const src = entries[i]
            const entryId = src.entryId !== undefined
                ? src.entryId
                : (transcriptEntryIdSeq = transcriptEntryIdSeq + 1, transcriptEntryIdSeq)
            preparedEntries.push(Object.assign({ entryId: entryId }, src))
        }
        if (preparedEntries.length === 0) {
            return
        }
        const nextEntries = transcriptEntries.concat(preparedEntries)
        const overflow = nextEntries.length - maxTranscriptEntries
        transcriptEntries = overflow > 0 ? nextEntries.slice(overflow) : nextEntries
        Qt.callLater(function() {
            if (!transcript) {
                return
            }
            if (shouldStickToBottom) {
                transcript.positionViewAtEnd()
                return
            }
            const nowViewportHeight = Math.max(0, Number(transcript.height || 0))
            const nowContentHeight = Math.max(0, Number(transcript.contentHeight || 0))
            const nowMaxContentY = Math.max(0, nowContentHeight - nowViewportHeight)
            transcript.contentY = Math.max(0, Math.min(nowMaxContentY, previousContentY))
        })
    }

    function appendTranscriptEntry(entry) {
        const isMessageEntry = entry && entry.kind === "message"
        appendTranscriptEntries([entry], isMessageEntry)
    }

    function keepPromptCursorVisible() {
        if (!promptInput || !promptInputScroll) {
            return
        }
        const flickable = promptInputScroll.contentItem
        if (!flickable) {
            return
        }
        const caretTop = Number(promptInput.cursorRectangle.y || 0)
        const caretHeight = Number(promptInput.cursorRectangle.height || 0)
        const caretBottom = caretTop + caretHeight
        const viewportTop = Number(flickable.contentY || 0)
        const viewportHeight = Math.max(1, Number(flickable.height || promptInputScroll.height || 0))
        const viewportBottom = viewportTop + viewportHeight
        if (caretBottom > viewportBottom) {
            flickable.contentY = Math.max(0, caretBottom - viewportHeight)
        } else if (caretTop < viewportTop) {
            flickable.contentY = Math.max(0, caretTop)
        }
    }

    function applyInitialLoadAutoScrollIfNeeded() {
        if (!initialLoadAutoScrollPending) {
            return
        }
        initialLoadAutoScrollPending = false
        Qt.callLater(function() {
            if (!transcript || Number(transcript.count || 0) <= 0) {
                return
            }
            transcript.positionViewAtEnd()
            Qt.callLater(function() {
                if (transcript && Number(transcript.count || 0) > 0) {
                    transcript.positionViewAtEnd()
                }
            })
        })
    }

    function isToolEntryExpanded(entryId) {
        return expandedToolEntries[String(entryId)] === true
    }

    function toggleToolEntryExpanded(entryId) {
        const key = String(entryId)
        const nextValue = !isToolEntryExpanded(entryId)
        // Assign to the QML property first so the new object is immediately
        // GC-rooted, then write the key — same fix as the pending property.
        expandedToolEntries = Object.assign({}, expandedToolEntries)
        expandedToolEntries[key] = nextValue
    }

    function appendDebugStatus(text) {
        if (debugEnabled) {
            appendMessage("tool", text)
        }
    }

    function ensureConversation(onReady) {
        if (conversationId.length > 0) {
            if (onReady) {
                onReady()
            }
            return
        }

        const command = helperCommand("ensure --title " + shellEscape(conversationTitle))
        appendDebugStatus("Initializing conversation…")
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus("Conversation init failed: " + payload.error)
                    return
                }
                conversationId = payload.conversation_id
                appendStatus("Using conversation " + conversationId)
                reloadConversationList()
                refreshConversation()
                if (onReady) {
                    onReady()
                }
            },
            function(stderr) {
                appendStatus("Conversation error: " + stderr)
            }
        )
    }

    function newConversation() {
        if (busy) {
            return
        }

        const title = conversationTitle + " " + Date.now()
        const command = helperCommand("create --title " + shellEscape(title))
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus("Failed to create new conversation: " + payload.error)
                    return
                }

                conversationId = payload.conversation_id
                promptText = ""
                currentMessageCount = 0
                expandedToolEntries = ({})
                transcriptEntries = [
                    {
                        kind: "status",
                        role: "status",
                        text: "New conversation ready",
                    }
                ]
                appendStatus("Using conversation " + conversationId)
                reloadConversationList()
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    function conversationIndexById(id) {
        for (let i = 0; i < conversationChoices.length; i++) {
            if (conversationChoices[i].id === id) {
                return i
            }
        }
        return -1
    }

    function reloadConversationList(onLoaded) {
        const command = helperCommand("list --max-age-days " + maxSessionAgeDays())
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus(payload.error)
                    return
                }

                const conversations = payload.conversations || []
                conversationChoices = conversations.map(function(conversation) {
                    const timestamp = String(conversation.updated_at || "").trim()
                    const titleText = timestamp.length > 0
                        ? (conversation.title + " · " + timestamp)
                        : conversation.title
                    return {
                        id: conversation.id,
                        title: titleText + " (" + conversation.message_count + ")",
                    }
                })

                const idx = conversationIndexById(conversationId)
                if (idx >= 0 && !panelMode) {
                    conversationPicker.currentIndex = idx
                }

                if (onLoaded) {
                    onLoaded(conversations)
                }
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    function deleteConversation(targetId) {
        if (busy || targetId.length === 0) {
            return
        }

        const deletingCurrent = targetId === conversationId
        const command = helperCommand("delete " + shellEscape(targetId))
        runCommand(
            command,
            function(stdout) {
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus("Failed to delete conversation: " + payload.error)
                    return
                }

                appendStatus("Deleted conversation " + targetId)
                reloadConversationList(function(conversations) {
                    if (!deletingCurrent) {
                        return
                    }

                    if (conversations.length > 0) {
                        conversationId = conversations[0].id
                        conversationPicker.currentIndex = 0
                        refreshConversation()
                        return
                    }

                    conversationId = ""
                    promptText = ""
                    currentMessageCount = 0
                    expandedToolEntries = ({})
                    transcriptEntries = [
                        {
                            kind: "status",
                            role: "status",
                            text: "No conversations yet",
                        }
                    ]
                })
            },
            function(stderr) {
                appendStatus(stderr)
            }
        )
    }

    function switchConversation(index) {
        if (index < 0 || index >= conversationChoices.length) {
            return
        }

        const selectedId = conversationChoices[index].id
        if (selectedId === conversationId) {
            return
        }

        conversationId = selectedId
        promptText = ""
        currentMessageCount = 0
        expandedToolEntries = ({})
        transcriptEntries = []
        appendStatus("Switched to conversation " + conversationId)
        refreshConversation()
    }

    function loadSelectedConversation() {
        if (busy || loadingConversation) {
            return
        }

        const idx = conversationPicker.currentIndex
        if (idx < 0 || idx >= conversationChoices.length) {
            appendStatus("No conversation selected")
            return
        }

        const selectedId = conversationChoices[idx].id
        if (selectedId !== conversationId) {
            switchConversation(idx)
            return
        }

        refreshConversation()
    }

    function refreshConversation() {
        if (conversationId.length === 0 || loadingConversation) {
            return
        }

        const baselineCount = Math.max(0, Number(currentMessageCount || 0))
        const useIncrementalLoad = baselineCount > 0
        const requestId = conversationId
        conversationLoadSequence = conversationLoadSequence + 1
        const sequence = conversationLoadSequence
        loadingConversation = true
        loadingConversationStartedAtMs = Date.now()
        conversationLoadTimeoutTimer.stop()
        conversationLoadTimeoutTimer.start()
        if (!hasRealMessages) {
            appendStatus("Loading conversation…")
        }

        // NOTE: `--tail` is a UI-only fetch limit. Core conversation context remains
        // intact in the daemon store for model prompting.
        // Role filtering is done server-side: include tool messages only when
        // debug mode is on so they count against the tail budget correctly.
        const roles = debugEnabled ? "user,assistant,tool" : "user,assistant"
        const command = useIncrementalLoad
            ? helperCommand("get " + shellEscape(requestId) + " --after-count " + baselineCount + " --roles " + shellEscape(roles))
            : helperCommand("get " + shellEscape(requestId) + " --tail " + maxRenderedMessages() + " --roles " + shellEscape(roles))
        runCommand(
            command,
            function(stdout) {
                if (sequence !== conversationLoadSequence) {
                    return
                }
                conversationLoadTimeoutTimer.stop()
                loadingConversation = false
                loadingConversationStartedAtMs = 0
                const payload = JSON.parse(stdout)
                if (payload.error) {
                    appendStatus(payload.error)
                    return
                }

                if (requestId !== conversationId) {
                    return
                }

                const allMessages = payload.messages || []
                const payloadCount = Number(payload.message_count || allMessages.length)
                const newEntries = []
                for (let i = 0; i < allMessages.length; i++) {
                    const message = allMessages[i]
                    const entry = buildMessageEntry(message.role, clipLoadedMessageText(message.content), {
                        historicalLoad: true,
                    })
                    if (entry) {
                        newEntries.push(entry)
                    }
                }

                if (useIncrementalLoad) {
                    if (payloadCount < baselineCount) {
                        currentMessageCount = payloadCount
                        return
                    }
                    appendTranscriptEntries(newEntries, true)
                    currentMessageCount = payloadCount
                    applyInitialLoadAutoScrollIfNeeded()
                    return
                }

                // Bootstrap path: populate visible history once from tail.
                expandedToolEntries = ({})
                transcriptEntryIdSeq = 0
                transcriptEntries = []
                appendTranscriptEntries(newEntries, false)
                currentMessageCount = payloadCount
                if (payload.truncated) {
                    appendStatus("Showing latest " + allMessages.length + " of " + currentMessageCount + " messages")
                }
                if (currentMessageCount === 0) {
                    appendStatus("No messages yet")
                }
                applyInitialLoadAutoScrollIfNeeded()
            },
            function(stderr) {
                if (sequence !== conversationLoadSequence) {
                    return
                }
                conversationLoadTimeoutTimer.stop()
                loadingConversation = false
                loadingConversationStartedAtMs = 0
                appendStatus(stderr)
            }
        )
    }

    function clearTranscriptView() {
        expandedToolEntries = ({})
        transcriptEntries = [
            {
                kind: "status",
                role: "status",
                text: "View cleared",
            }
        ]
    }

    function sendPrompt(textValue) {
        const sourceText = (textValue === undefined || textValue === null) ? promptText : textValue
        const prompt = sourceText.trim()
        if (prompt.length === 0 || busy) {
            return
        }

        ensureConversation(function() {
            busy = true
            lateResponsePollTimer.stop()
            lateResponsePollRemaining = 0
            promptText = ""

            const sendCommand = helperCommand("send " + shellEscape(conversationId) + " " + shellEscape(prompt))
            runCommand(
                sendCommand,
                function(sendOut) {
                    const payload = JSON.parse(sendOut)
                    if (payload.error) {
                        busy = false
                        appendStatus(payload.error)
                        return
                    }
                    refreshConversation()
                    reloadConversationList()
                    const startAwait = function(initialCount) {
                        const awaitCommand = helperCommand(
                            "await "
                            + shellEscape(conversationId)
                            + " --initial-count "
                            + initialCount
                            + " --timeout "
                            + awaitTimeoutSeconds
                        )
                        runCommand(
                            awaitCommand,
                            function(awaitOut) {
                                busy = false
                                const awaitPayload = JSON.parse(awaitOut)
                                if (awaitPayload.error) {
                                    appendStatus(awaitPayload.error)
                                }
                                refreshConversation()
                                reloadConversationList()
                                lateResponsePollRemaining = panelMode ? 6 : 12
                                lateResponsePollTimer.start()
                            },
                            function(awaitErr) {
                                busy = false
                                appendStatus(awaitErr)
                                refreshConversation()
                                reloadConversationList()
                                lateResponsePollRemaining = panelMode ? 6 : 12
                                lateResponsePollTimer.start()
                            }
                        )
                    }

                    const countCommand = helperCommand("get " + shellEscape(conversationId) + " --after-count 2147483647")
                    runCommand(
                        countCommand,
                        function(countOut) {
                            const countPayload = JSON.parse(countOut)
                            if (countPayload.error) {
                                startAwait(currentMessageCount)
                                return
                            }
                            currentMessageCount = Number(countPayload.message_count || currentMessageCount)
                            startAwait(currentMessageCount)
                        },
                        function(_countErr) {
                            startAwait(currentMessageCount)
                        }
                    )
                },
                function(sendErr) {
                    busy = false
                    appendStatus(sendErr)
                }
            )
        })
    }

    Plasma5Support.DataSource {
        id: executable
        engine: "executable"
        connectedSources: []

        onNewData: function(sourceName, data) {
            const callbacks = pending[sourceName]
            if (!callbacks) {
                disconnectSource(sourceName)
                return
            }

            const exitCode = data["exit code"]
            const stdout = (data.stdout || "").trim()
            const stderr = (data.stderr || "").trim()

            try {
                if (exitCode === 0) {
                    if (callbacks.debug) {
                        appendToolExecutionDebug("ok", sourceName, stdout)
                    }
                    callbacks.success(stdout)
                } else {
                    if (callbacks.debug) {
                        appendToolExecutionDebug("error", sourceName, stderr.length > 0 ? stderr : stdout)
                    }
                    callbacks.error(stderr.length > 0 ? stderr : stdout)
                }
            } catch (callbackError) {
                appendStatus("Widget callback error: " + callbackError)
                busy = false
            } finally {
                delete pending[sourceName]
                disconnectSource(sourceName)
            }
        }
    }

    Timer {
        id: conversationLoadTimeoutTimer
        interval: root.conversationLoadTimeoutMs
        repeat: false
        running: false
        onTriggered: {
            if (!loadingConversation) {
                return
            }
            loadingConversation = false
            loadingConversationStartedAtMs = 0
            appendStatus("Loading timed out. Please try Refresh.")
        }
    }

    Timer {
        id: startupTimer
        interval: 250
        repeat: false
        running: false
        onTriggered: {
            if (!root.hideWidget) {
                reloadConversationList(function(conversations) {
                    if (conversations.length > 0) {
                        conversationId = conversations[0].id
                        conversationPicker.currentIndex = 0
                        appendStatus("Conversation selected. Click Load to open history.")
                        return
                    }
                    appendStatus("No conversations yet. Send a message to start.")
                })
            }
        }
    }

    Timer {
        id: servicePollTimer
        interval: 5000
        repeat: true
        running: true
        onTriggered: {
            if (root.loadingConversation && root.loadingConversationStartedAtMs > 0) {
                const elapsedMs = Date.now() - root.loadingConversationStartedAtMs
                if (elapsedMs >= root.conversationLoadTimeoutMs + 1000) {
                    root.loadingConversation = false
                    root.loadingConversationStartedAtMs = 0
                    root.appendStatus("Loading timed out. Please try Refresh.")
                }
            }
            refreshServiceStatus(undefined, {
                silent: true,
            })
        }
    }

    Timer {
        id: lateResponsePollTimer
        interval: 5000
        repeat: true
        running: false
        onTriggered: {
            if (busy || conversationId.length === 0) {
                return
            }
            refreshConversation()
            reloadConversationList()
            lateResponsePollRemaining = Math.max(0, lateResponsePollRemaining - 1)
            if (lateResponsePollRemaining <= 0) {
                stop()
            }
        }
    }

    Shortcut {
        sequence: "Ctrl++"
        context: Qt.WindowShortcut
        onActivated: root.zoomInUi()
    }

    Shortcut {
        sequence: "Ctrl+="
        context: Qt.WindowShortcut
        onActivated: root.zoomInUi()
    }

    Shortcut {
        sequence: "Ctrl+-"
        context: Qt.WindowShortcut
        onActivated: root.zoomOutUi()
    }

    Shortcut {
        sequence: "Ctrl+_"
        context: Qt.WindowShortcut
        onActivated: root.zoomOutUi()
    }

    Shortcut {
        sequence: "Ctrl+0"
        context: Qt.WindowShortcut
        onActivated: root.resetZoomUi()
    }

    Rectangle {
        anchors.fill: parent
        visible: !root.hideWidget
        color: root.themeBackgroundColor
        border.width: 1
        border.color: root.themeDisabledTextColor
        radius: 8
    }

    ColumnLayout {
        visible: !root.hideWidget
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            Image {
                source: root.busy
                    ? Qt.resolvedUrl("../images/adele_thinking.png")
                    : Qt.resolvedUrl("../images/adele.png")
                sourceSize.width: root.scaledTopIconSize
                sourceSize.height: root.scaledTopIconSize
                fillMode: Image.PreserveAspectFit
                Layout.preferredWidth: root.scaledTopIconSize
                Layout.preferredHeight: root.scaledTopIconSize
            }

            QQC2.Label {
                text: root.activeService === root.developmentService ? "Adele (Dev)" : "Adele"
                font.bold: true
                color: root.themeTextColor
                Layout.fillWidth: true
            }
        }

        Flow {
            id: conversationControls
            Layout.fillWidth: true
            spacing: 6

            QQC2.ComboBox {
                id: conversationPicker
                visible: true
                width: {
                    const buttonWidths = loadButton.implicitWidth
                        + refreshListButton.implicitWidth + conversationControls.spacing
                        + (settingsButton.visible ? (settingsButton.implicitWidth + conversationControls.spacing) : 0)
                    return Math.max(root.ultraNarrow ? 140 : 180, conversationControls.width - buttonWidths - conversationControls.spacing)
                }
                enabled: !root.busy && !root.loadingConversation
                model: root.conversationChoices
                textRole: "title"
                delegate: QQC2.ItemDelegate {
                    id: conversationDelegate
                    required property var modelData
                    required property int index
                    width: conversationPicker.width
                    highlighted: conversationPicker.highlightedIndex === index
                    background: Rectangle {
                        color: conversationDelegate.highlighted
                            ? root.themeHighlightColor
                            : "transparent"
                    }

                    contentItem: RowLayout {
                        spacing: 6

                        QQC2.Label {
                            Layout.fillWidth: true
                            text: modelData.title
                            color: conversationDelegate.highlighted
                                ? root.themeHighlightedTextColor
                                : root.themeTextColor
                            elide: Text.ElideRight
                        }

                        QQC2.ToolButton {
                            icon.name: "edit-delete"
                            display: QQC2.AbstractButton.IconOnly
                            enabled: !root.busy
                            onClicked: {
                                root.deleteConversation(modelData.id)
                                conversationPicker.popup.close()
                            }
                        }
                    }

                    onClicked: {
                        const idx = root.conversationIndexById(modelData.id)
                        conversationPicker.currentIndex = idx
                        conversationPicker.popup.close()
                        switchConversation(idx)
                    }
                }
                onActivated: function(index) {
                    switchConversation(index)
                }
            }

            QQC2.Button {
                id: loadButton
                text: loadingConversation ? "Loading…" : "Load"
                enabled: !busy && !loadingConversation
                onClicked: panelMode ? refreshServiceStatus() : loadSelectedConversation()
            }

            QQC2.Button {
                id: refreshListButton
                text: "Refresh List"
                enabled: !busy && !loadingConversation
                onClicked: reloadConversationList()
            }

            QQC2.Button {
                id: settingsButton
                visible: panelMode && !root.ultraNarrow
                text: "Settings"
                enabled: !busy
                onClicked: openSettingsDialog()
            }
        }

        QQC2.ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                id: transcript
                model: root.transcriptEntries
                spacing: 6
                clip: true

                header: Column {
                    width: transcript.width
                    visible: !root.hasRealMessages
                    spacing: 8
                    topPadding: 40

                    Image {
                        source: Qt.resolvedUrl("../images/adele.png")
                        sourceSize.width: root.scaledHeaderIconSize
                        sourceSize.height: root.scaledHeaderIconSize
                        width: root.scaledHeaderIconSize
                        height: root.scaledHeaderIconSize
                        fillMode: Image.PreserveAspectFit
                        anchors.horizontalCenter: parent.horizontalCenter
                    }

                    QQC2.Label {
                        text: "Hi! I'm Adele! Ask me anything..."
                        font.pointSize: root.baseFontPointSize * root.uiScale
                        color: root.themeDisabledTextColor
                        anchors.horizontalCenter: parent.horizontalCenter
                    }
                }

                onCountChanged: {
                    // Intentionally no auto-scroll.
                }

                delegate: Item {
                    required property var modelData
                    readonly property bool isStatus: modelData.kind === "status"
                    readonly property bool isTool: modelData.role === "tool"
                    readonly property string toolName: String(modelData.toolName || "Tool")
                    readonly property bool isAssistant: modelData.role === "assistant"
                    readonly property string messageBodyText: String(modelData.text || "")
                    readonly property bool renderAssistantAsMarkdown: root.shouldRenderAssistantAsMarkdown(messageBodyText)
                    readonly property bool toolExpanded: root.isToolEntryExpanded(modelData.entryId)
                    readonly property real avatarSize: root.transcriptAvatarSize
                    readonly property real bubbleWidth: (isStatus || isTool)
                        ? root.transcriptWideBubbleWidth
                        : root.transcriptMessageBubbleWidth
                    readonly property var avatarSources: isAssistant ? [root.adeleAvatarSource] : root.userAvatarCandidates()

                    visible: !(isStatus || isTool) || root.debugEnabled
                    width: ListView.view.width
                    implicitHeight: visible ? rowContainer.implicitHeight + 2 : 0

                    RowLayout {
                        id: rowContainer
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        spacing: 6
                        layoutDirection: (isStatus || isTool) ? Qt.LeftToRight : (isAssistant ? Qt.RightToLeft : Qt.LeftToRight)

                        Item {
                            Layout.preferredWidth: (isStatus || isTool) ? 0 : avatarSize
                            Layout.preferredHeight: (isStatus || isTool) ? 0 : avatarSize
                            Layout.alignment: Qt.AlignTop
                            visible: !(isStatus || isTool)

                            Rectangle {
                                anchors.fill: parent
                                radius: width / 2
                                color: root.themeBackgroundColor
                                border.width: 1
                                border.color: root.themeDisabledTextColor
                                clip: true

                                Image {
                                    id: avatarImage
                                    property int candidateIndex: 0
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.top: parent.top
                                    width: isAssistant ? parent.width * 1.9 : parent.width
                                    height: isAssistant ? parent.height * 1.9 : parent.height
                                    fillMode: Image.PreserveAspectCrop
                                    horizontalAlignment: Image.AlignHCenter
                                    verticalAlignment: isAssistant ? Image.AlignTop : Image.AlignVCenter
                                    source: avatarSources.length > 0 ? avatarSources[Math.min(candidateIndex, avatarSources.length - 1)] : ""
                                    visible: status === Image.Ready

                                    onStatusChanged: {
                                        if (status === Image.Error && candidateIndex < avatarSources.length - 1) {
                                            candidateIndex += 1
                                        }
                                    }
                                }

                                Kirigami.Icon {
                                    anchors.fill: parent
                                    source: isAssistant ? "preferences-desktop-user" : "user-identity"
                                    visible: !avatarImage.visible
                                }
                            }
                        }

                        Rectangle {
                            id: bubble
                            Layout.fillWidth: true
                            Layout.maximumWidth: bubbleWidth
                            Layout.alignment: Qt.AlignTop
                            Layout.preferredWidth: bubbleWidth
                            implicitWidth: Layout.preferredWidth
                            implicitHeight: bubbleContent.implicitHeight + 12
                            height: implicitHeight
                            radius: isStatus ? 0 : 8
                            color: isStatus
                                ? "transparent"
                                : (isTool
                                    ? root.themeBackgroundColor
                                    : (isAssistant ? root.themeBackgroundColor : root.themeHighlightColor))
                            border.width: isStatus ? 0 : 1
                            border.color: root.themeDisabledTextColor

                            ColumnLayout {
                                id: bubbleContent
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.margins: 6
                                spacing: 4

                                QQC2.Button {
                                    visible: isTool
                                    Layout.fillWidth: true
                                    text: (toolExpanded ? "▾" : "▸") + " " + toolName
                                    onClicked: root.toggleToolEntryExpanded(modelData.entryId)
                                }

                            TextEdit {
                                id: messageText
                                Layout.fillWidth: true
                                Layout.preferredHeight: visible ? contentHeight : 0
                                visible: !isTool || toolExpanded
                                readOnly: true
                                selectByMouse: true
                                selectByKeyboard: true
                                wrapMode: TextEdit.Wrap
                                textFormat: (modelData.kind === "message" && isAssistant && renderAssistantAsMarkdown)
                                    ? Text.MarkdownText
                                    : Text.PlainText
                                text: isStatus
                                    ? "[status] " + messageBodyText
                                    : messageBodyText
                                color: isStatus
                                    ? root.themeDisabledTextColor
                                    : (isTool
                                        ? root.themeTextColor
                                        : (isAssistant
                                        ? root.themeTextColor
                                        : root.themeHighlightedTextColor))
                                font.pointSize: root.baseFontPointSize * root.uiScale
                                font.italic: isStatus
                                font.bold: false
                                activeFocusOnPress: true
                                selectedTextColor: (isAssistant || isTool) ? root.themeHighlightedTextColor : root.themeTextColor
                                selectionColor: (isAssistant || isTool) ? root.themeHighlightColor : root.themeBackgroundColor
                                onLinkActivated: function(link) {
                                    Qt.openUrlExternally(link)
                                }
                            }
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true

            QQC2.ScrollView {
                id: promptInputScroll
                Layout.fillWidth: true
                Layout.preferredHeight: Math.round(72 * root.uiScale)
                Layout.maximumHeight: Math.round(180 * root.uiScale)
                clip: true
                QQC2.ScrollBar.horizontal.policy: QQC2.ScrollBar.AlwaysOff
                QQC2.ScrollBar.vertical.policy: QQC2.ScrollBar.AsNeeded

                QQC2.TextArea {
                    id: promptInput
                    width: promptInputScroll.availableWidth
                    placeholderText: "Ask Adele…"
                    wrapMode: TextEdit.Wrap
                    text: root.promptText
                    enabled: !busy
                    onTextChanged: {
                        root.promptText = text
                        Qt.callLater(root.keepPromptCursorVisible)
                    }
                    onCursorPositionChanged: Qt.callLater(root.keepPromptCursorVisible)
                    onActiveFocusChanged: {
                        if (activeFocus) {
                            Qt.callLater(root.keepPromptCursorVisible)
                        }
                    }

                    Keys.onPressed: function(event) {
                        const isEnterKey = event.key === Qt.Key_Return || event.key === Qt.Key_Enter
                        if (!isEnterKey) {
                            return
                        }

                        if (event.modifiers & Qt.MetaModifier) {
                            insert(cursorPosition, "\n")
                            event.accepted = true
                            return
                        }

                        sendPrompt(text)
                        event.accepted = true
                    }
                }
            }
        }

        Flow {
            id: actionControls
            Layout.fillWidth: true
            spacing: 6

            QQC2.Button {
                id: sendButton
                text: busy ? "…" : "Send"
                enabled: !busy
                onClicked: sendPrompt(root.promptText)
            }

            QQC2.Button {
                text: "New"
                enabled: !busy
                onClicked: newConversation()
            }

            QQC2.Button {
                text: "Refresh"
                visible: !panelMode
                enabled: !busy
                onClicked: refreshConversation()
            }

            QQC2.Button {
                text: "Clear"
                enabled: !busy
                onClicked: clearTranscriptView()
            }

            QQC2.ComboBox {
                id: servicePicker
                visible: root.devServiceRunning && !root.ultraNarrow
                width: 170
                model: root.serviceChoices
                textRole: "label"
                onActivated: function(index) {
                    switchService(index)
                }
            }

            QQC2.CheckBox {
                id: debugCheckBox
                visible: !root.ultraNarrow
                text: "Debug"
                checked: root.debugEnabled
                contentItem: QQC2.Label {
                    text: debugCheckBox.text
                    color: root.themeTextColor
                    verticalAlignment: Text.AlignVCenter
                    leftPadding: debugCheckBox.indicator && debugCheckBox.indicator.visible
                        ? debugCheckBox.indicator.width + debugCheckBox.spacing
                        : 0
                }
                onToggled: {
                    root.debugEnabled = checked
                    if (root.debugEnabled && root.conversationId.length > 0 && !root.loadingConversation) {
                        root.refreshConversation()
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        expandedToolEntries = ({})
        transcriptEntries = [
            {
                kind: "status",
                role: "status",
                text: "Ready",
            }
        ]
        appendDebugStatus("Widget loaded")
        refreshServiceStatus(function() {
            if (panelMode) {
                ensureConversation()
            } else {
                startupTimer.start()
            }
        })
    }
}
