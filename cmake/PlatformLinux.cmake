# =========================================================
# Linux part of the build (included from CMakeLists.txt)
# =========================================================

# ---------------------------------------------------------
# 1) Before dependencies are searched
# ---------------------------------------------------------
macro(nbmx_platform_setup)
    # Install into <project>/install/AppDir/usr.
    # Also replaces the /usr/local default left in older build caches.
    # Override with -DCMAKE_INSTALL_PREFIX=... or "cmake --install build --prefix ...".
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
        set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/AppDir/usr" CACHE PATH "Install prefix" FORCE)
    endif()

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
    find_package(SDL2 REQUIRED QUIET)
    find_package(gmsh REQUIRED)
endmacro()

# ---------------------------------------------------------
# 2) After the executable is created
# ---------------------------------------------------------
macro(nbmx_platform_target)
    # Static NVRTC: no nvrtc / nvrtc-builtins .so to ship.
    # nvrtc_static depends on the other two, so they are attached to it (right link order).
    foreach(_lib nvrtc_static nvrtc-builtins_static nvptxcompiler_static)
        if(NOT EXISTS "${CUDAToolkit_LIBRARY_DIR}/lib${_lib}.a")
            message(FATAL_ERROR "Missing ${CUDAToolkit_LIBRARY_DIR}/lib${_lib}.a")
        endif()
    endforeach()

    add_library(nbmx_nvptxcompiler_static STATIC IMPORTED)
    set_target_properties(nbmx_nvptxcompiler_static PROPERTIES
            IMPORTED_LOCATION "${CUDAToolkit_LIBRARY_DIR}/libnvptxcompiler_static.a")
    add_library(nbmx_nvrtc_builtins_static STATIC IMPORTED)
    set_target_properties(nbmx_nvrtc_builtins_static PROPERTIES
            IMPORTED_LOCATION "${CUDAToolkit_LIBRARY_DIR}/libnvrtc-builtins_static.a")
    add_library(nbmx_nvrtc_static STATIC IMPORTED)
    set_target_properties(nbmx_nvrtc_static PROPERTIES
            IMPORTED_LOCATION "${CUDAToolkit_LIBRARY_DIR}/libnvrtc_static.a"
            INTERFACE_LINK_LIBRARIES "nbmx_nvrtc_builtins_static;nbmx_nvptxcompiler_static")

    target_include_directories(${PROJECT_NAME} PRIVATE
            ${SDL2_INCLUDE_DIRS}
            ${gmsh_SOURCE_DIR}/api ${gmsh_BINARY_DIR}/api)

    target_link_libraries(${PROJECT_NAME} PRIVATE
            $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2>
            PkgConfig::FFMPEG
            gmsh
            CUDA::cudart_static          # static CUDA runtime
            nbmx_nvrtc_static            # static NVRTC (+ builtins + ptx compiler)
            stdc++fs
            ${CMAKE_DL_LIBS}
            rt)

    # libstdc++/libgcc statically -> fewer GLIBCXX version problems on the user's distro
    target_link_options(${PROJECT_NAME} PRIVATE -static-libstdc++ -static-libgcc)

    # Bundled .so files live in usr/lib next to usr/bin
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

    # Install the executable + the .so files it needs (minus system / driver libraries)
    install(TARGETS ${PROJECT_NAME}
            RUNTIME_DEPENDENCY_SET nbmx_deps
            RUNTIME DESTINATION bin)

    install(RUNTIME_DEPENDENCY_SET nbmx_deps
            DESTINATION lib
            PRE_EXCLUDE_REGEXES
            # Core system
            "ld-linux.*"
            "libc\\.so.*"
            "libm\\.so.*"
            "libpthread\\.so.*"
            "libdl\\.so.*"
            "librt\\.so.*"
            "libgcc_s\\.so.*"
            "libstdc\\+\\+\\.so.*"
            # NVIDIA driver / OpenGL / EGL / Vulkan (must come from the user's driver)
            "libcuda\\.so.*"
            "libnvidia-.*"
            "libOpenGL\\.so.*"
            "libGL\\.so.*"
            "libGLX.*"
            "libGLdispatch\\.so.*"
            "libEGL.*"
            "libGLESv2\\.so.*"
            "libvulkan\\.so.*"
            # X11
            "libX11\\.so.*"
            "libXext\\.so.*"
            "libXi\\.so.*"
            "libXcursor\\.so.*"
            "libXinerama\\.so.*"
            "libXrandr\\.so.*"
            "libXss\\.so.*"
            "libXxf86vm\\.so.*"
            "libXrender\\.so.*"
            "libX11-xcb\\.so.*"
            "libxcb.*"
            "libXfixes\\.so.*"
            # Wayland
            "libwayland-.*"
            "libxkbcommon\\.so.*"
            "libdecor-.*"
            # DRM / VAAPI
            "libdrm\\.so.*"
            "libgbm\\.so.*"
            "libva\\.so.*"
            "libva-drm\\.so.*"
            "libva-x11\\.so.*"
            "libvdpau\\.so.*"
            # Audio
            "libpulse\\.so.*"
            "libpulsecommon.*"
            "libasound\\.so.*"
            # Optional system services
            "libdbus-1\\.so.*"
            "libsystemd\\.so.*"
            "libapparmor\\.so.*"
            # Fonts
            "libfontconfig\\.so.*"
            "libXft\\.so.*")

    # Icon in the standard location (desktop menus)
    if(EXISTS "${APP_ICON_PNG}")
        install(FILES "${APP_ICON_PNG}"
                DESTINATION share/icons/hicolor/256x256/apps)
    endif()
endmacro()

# ---------------------------------------------------------
# 3) Packaging: AppImage, built entirely by CMake
#   cmake --install  ->  install/AppDir/usr/...
#                    ->  AppDir/AppRun, .desktop, icon, .DirIcon
#                    ->  appimagetool  ->  install/NewtonBioMorphX-<ver>-<arch>.AppImage
#   Turn off with -DBUILD_APPIMAGE=OFF.
# ---------------------------------------------------------
macro(nbmx_platform_package)
    option(BUILD_APPIMAGE "Create the AppImage at the end of cmake --install" ON)
    set(NBMX_ARCH "${CMAKE_SYSTEM_PROCESSOR}")   # x86_64 / aarch64

    if(BUILD_APPIMAGE)
        # appimagetool: use one from PATH, otherwise download it once into the build folder
        find_program(APPIMAGETOOL_EXECUTABLE NAMES appimagetool appimagetool-${NBMX_ARCH}.AppImage)
        if(NOT APPIMAGETOOL_EXECUTABLE)
            set(_tool "${CMAKE_BINARY_DIR}/tools/appimagetool-${NBMX_ARCH}.AppImage")
            if(NOT EXISTS "${_tool}")
                message(STATUS "Downloading appimagetool ...")
                file(DOWNLOAD
                        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${NBMX_ARCH}.AppImage"
                        "${_tool}" STATUS _dl_status)
                list(GET _dl_status 0 _dl_code)
                if(NOT _dl_code EQUAL 0)
                    file(REMOVE "${_tool}")
                    message(WARNING "appimagetool download failed (${_dl_status}). "
                            "Install it on PATH or pass -DAPPIMAGETOOL_EXECUTABLE=...")
                endif()
            endif()
            if(EXISTS "${_tool}")
                file(CHMOD "${_tool}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
                set(APPIMAGETOOL_EXECUTABLE "${_tool}" CACHE FILEPATH "appimagetool" FORCE)
            endif()
        endif()

        if(NOT APPIMAGETOOL_EXECUTABLE)
            message(WARNING "No appimagetool -> install will not create an AppImage")
        elseif(NOT EXISTS "${APP_ICON_PNG}")
            message(WARNING "AppImage needs an icon: ${APP_ICON_PNG} not found -> no AppImage")
        else()
            message(STATUS "AppImage: ${APPIMAGETOOL_EXECUTABLE}")
            set(_ai "${CMAKE_BINARY_DIR}/appimage")

            # AppRun: entry point of the AppImage, starts the real executable
            file(WRITE "${_ai}/AppRun"
                    "#!/bin/sh
HERE=\"$(dirname \"$(readlink -f \"$0\")\")\"
exec \"$HERE/usr/bin/${PROJECT_NAME}\" \"$@\"
")

            # Desktop entry (name, icon, menu category)
            file(WRITE "${_ai}/${PROJECT_NAME}.desktop"
                    "[Desktop Entry]
Type=Application
Name=${PROJECT_NAME}
Exec=${PROJECT_NAME}
Icon=${PROJECT_NAME}
StartupWMClass=${PROJECT_NAME}
Categories=Science;
Terminal=false
")

            # Install-time script. @VARS@ are filled in now, ${VARS} at install time
            # (so the AppDir follows the real install prefix, also with --prefix).
            file(CONFIGURE OUTPUT "${_ai}/make_appimage.cmake" @ONLY CONTENT [[
get_filename_component(_appdir "${CMAKE_INSTALL_PREFIX}" DIRECTORY)
get_filename_component(_outdir "${_appdir}" DIRECTORY)
message(STATUS "Preparing AppDir: ${_appdir}")

file(COPY "@_ai@/AppRun" DESTINATION "${_appdir}"
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
file(COPY "@_ai@/@PROJECT_NAME@.desktop" DESTINATION "${_appdir}")
file(COPY "@_ai@/@PROJECT_NAME@.desktop" DESTINATION "${CMAKE_INSTALL_PREFIX}/share/applications")
file(COPY_FILE "@APP_ICON_PNG@" "${_appdir}/@PROJECT_NAME@.png")
file(COPY_FILE "@APP_ICON_PNG@" "${_appdir}/.DirIcon")

# Show the RUNPATH so you can check that $ORIGIN/../lib is set
execute_process(COMMAND readelf -d "${CMAKE_INSTALL_PREFIX}/bin/@PROJECT_NAME@"
                OUTPUT_VARIABLE _readelf)
string(REGEX MATCH "[^\n]*(RUNPATH|RPATH)[^\n]*" _rpath "${_readelf}")
message(STATUS "${_rpath}")

set(_out "${_outdir}/@PROJECT_NAME@-@PROJECT_VERSION@-@NBMX_ARCH@.AppImage")
message(STATUS "Building ${_out} ...")
# APPIMAGE_EXTRACT_AND_RUN: lets appimagetool run on machines without FUSE
execute_process(
    COMMAND "@CMAKE_COMMAND@" -E env ARCH=@NBMX_ARCH@ APPIMAGE_EXTRACT_AND_RUN=1
            "@APPIMAGETOOL_EXECUTABLE@" "${_appdir}" "${_out}"
    WORKING_DIRECTORY "${_outdir}"
    RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "AppImage generation failed with code ${_result}")
endif()
message(STATUS "AppImage ready: ${_out}")
]])
            install(SCRIPT "${_ai}/make_appimage.cmake")
        endif()
    endif()
endmacro()