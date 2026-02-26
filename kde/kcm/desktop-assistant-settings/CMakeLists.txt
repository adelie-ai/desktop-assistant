cmake_minimum_required(VERSION 3.20)
project(desktop_assistant_kcm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(QT_DEFAULT_MAJOR_VERSION 6)

find_package(ECM REQUIRED NO_MODULE)
set(CMAKE_MODULE_PATH ${ECM_MODULE_PATH} ${CMAKE_MODULE_PATH})
include(KDEInstallDirs)
include(KDECMakeSettings)
include(KDECompilerSettings NO_POLICY_SCOPE)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" CACHE PATH "Library output directory" FORCE)

find_package(Qt6 REQUIRED COMPONENTS Core DBus)
find_package(KF6 REQUIRED COMPONENTS CoreAddons KCMUtils)

add_definitions(-DTRANSLATION_DOMAIN="kcm_desktopassistant")

kcmutils_add_qml_kcm(kcm_desktopassistant
    SOURCES
        desktopassistantkcm.cpp
    INSTALL_NAMESPACE
        plasma/kcms/systemsettings
)

install(FILES settings-applications-desktopassistant.desktop DESTINATION ${KDE_INSTALL_DATADIR}/systemsettings/categories)

target_link_libraries(kcm_desktopassistant
    PRIVATE
        Qt6::Core
        Qt6::DBus
        KF6::CoreAddons
        KF6::KCMUtils
)
