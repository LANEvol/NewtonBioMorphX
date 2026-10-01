# =========================================================
# Windows part of the build (included from CMakeLists.txt)
# =========================================================

# ---------------------------------------------------------
# 1) Before dependencies are searched
# ---------------------------------------------------------
macro(nbmx_platform_setup)
    # Avoid crashes in std::mutex when an older msvcp140.dll is loaded at runtime.
    # Set here so it also applies to libigl and all other targets.
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)

    # gmsh SDK location (-DGMSH_ROOT=... to override)
    set(GMSH_ROOT "C:/Users/jahanbak/repositories/gmsh-4.15.2-Windows64-sdk" CACHE PATH "gmsh SDK root")

    # Install into <project>/install (no admin rights needed).
    # Override with -DCMAKE_INSTALL_PREFIX=... or "cmake --install build --prefix ...".
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install prefix" FORCE)
    endif()

    find_package(FFMPEG REQUIRED)            # vcpkg's FindFFMPEG
    find_package(SDL2 CONFIG REQUIRED QUIET)
endmacro()

# ---------------------------------------------------------
# 2) After the executable is created
# ---------------------------------------------------------
macro(nbmx_platform_target)
    target_include_directories(${PROJECT_NAME} PRIVATE
            "${GMSH_ROOT}/include"
            ${FFMPEG_INCLUDE_DIRS})
    target_link_directories(${PROJECT_NAME} PRIVATE ${FFMPEG_LIBRARY_DIRS})

    target_link_libraries(${PROJECT_NAME} PRIVATE
            $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
            $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
            ${FFMPEG_LIBRARIES}
            "${GMSH_ROOT}/lib/gmsh.dll.lib"
            CUDA::nvrtc)                     # nvrtc64_*.dll (+ nvrtc-builtins, see below)

    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()

    # Icon: resource name GLFW_ICON -> Explorer/taskbar/shortcuts show it for the .exe,
    # and GLFW uses it automatically for the viewer window.
    if(EXISTS "${APP_ICON_ICO}")
        set(_app_rc "${CMAKE_BINARY_DIR}/app_icon.rc")
        file(WRITE "${_app_rc}" "GLFW_ICON ICON \"${APP_ICON_ICO}\"\n")
        target_sources(${PROJECT_NAME} PRIVATE "${_app_rc}")
    else()
        message(WARNING "Icon not found: ${APP_ICON_ICO} (the exe will use the default icon)")
    endif()

    # DLLs that are not found automatically:
    #   gmsh*.dll  : linked through the raw import lib, so CMake doesn't know where the DLL is
    #   nvrtc*.dll : nvrtc-builtins is loaded at runtime by NVRTC, the linker never sees it
    file(GLOB _gmsh_dlls
            "${GMSH_ROOT}/lib/gmsh*.dll"
            "${GMSH_ROOT}/bin/gmsh*.dll")
    file(GLOB _nvrtc_dlls
            "${CUDAToolkit_BIN_DIR}/nvrtc*.dll"
            "${CUDAToolkit_BIN_DIR}/x64/nvrtc*.dll")      # CUDA 13 layout
    list(FILTER _nvrtc_dlls EXCLUDE REGEX "\\.alt\\.dll$")
    if(NOT _gmsh_dlls)
        message(WARNING "No gmsh*.dll found in ${GMSH_ROOT}/lib or /bin")
    endif()
    if(NOT _nvrtc_dlls)
        message(WARNING "No nvrtc*.dll found in ${CUDAToolkit_BIN_DIR}")
    endif()

    # Copy them next to the exe after every build, so it runs from the build folder
    if(_gmsh_dlls OR _nvrtc_dlls)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${_gmsh_dlls} ${_nvrtc_dlls}
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                COMMENT "Copying gmsh and NVRTC DLLs next to ${PROJECT_NAME}.exe"
                VERBATIM)
    endif()

    # Install the exe + every DLL it links against (SDL2, VTK, FFmpeg, gmp, mpfr, ...).
    # RUNTIME_DEPENDENCIES must come right after the target, before RUNTIME DESTINATION.
    set(_dll_dirs
            "${CUDAToolkit_BIN_DIR}"
            "${CUDAToolkit_BIN_DIR}/x64"
            "${GMSH_ROOT}/lib"
            "${GMSH_ROOT}/bin")
    if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
        list(APPEND _dll_dirs "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
    endif()

    install(TARGETS ${PROJECT_NAME}
            RUNTIME_DEPENDENCIES
                PRE_EXCLUDE_REGEXES  "api-ms-.*" "ext-ms-.*" "nvcuda.*"   # nvcuda = NVIDIA driver, never ship it
                POST_EXCLUDE_REGEXES ".*[Ss][Yy][Ss][Tt][Ee][Mm]32.*"
                DIRECTORIES ${_dll_dirs}
            RUNTIME DESTINATION bin)
    install(FILES ${_nvrtc_dlls} ${_gmsh_dlls} DESTINATION bin)

    # MSVC runtime next to the exe (users never load an old msvcp140.dll)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION bin)
    include(InstallRequiredSystemLibraries)
endmacro()

# ---------------------------------------------------------
# 3) Packaging: portable ZIP (+ NSIS installer if NSIS is installed)
#    Runs automatically at the end of "cmake --install" (-DPACKAGE_ON_INSTALL=OFF to skip),
#    or on its own with: cmake --build build --config RelWithDebInfo --target package
# ---------------------------------------------------------
macro(nbmx_platform_package)
    set(CPACK_PACKAGE_NAME              "${PROJECT_NAME}")
    set(CPACK_PACKAGE_VENDOR            "Ebrahim Jahanbakhsh & Michel Milinkovitch")
    set(CPACK_PACKAGE_VERSION           "${PROJECT_VERSION}")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}")
    set(CPACK_PACKAGE_DIRECTORY         "${CMAKE_SOURCE_DIR}/install/packages")
    set(CPACK_PACKAGE_EXECUTABLES       "${PROJECT_NAME}" "${PROJECT_NAME}")   # Start-menu shortcut
    if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
        set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
    endif()

    set(CPACK_GENERATOR "ZIP")

    # NSIS doesn't add itself to PATH, so also look in its default folders and registry key
    set(_pf86 "ProgramFiles(x86)")
    find_program(MAKENSIS_EXECUTABLE makensis
            PATHS "$ENV{${_pf86}}/NSIS"
                  "$ENV{ProgramFiles}/NSIS"
                  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NSIS]"
                  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\NSIS]")
    if(MAKENSIS_EXECUTABLE)
        message(STATUS "NSIS found: ${MAKENSIS_EXECUTABLE} -> building ZIP + installer")
        list(APPEND CPACK_GENERATOR "NSIS")
        set(CPACK_NSIS_MODIFY_PATH OFF)
        set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
        if(EXISTS "${APP_ICON_ICO}")
            set(CPACK_NSIS_MUI_ICON    "${APP_ICON_ICO}")                    # installer .exe
            set(CPACK_NSIS_MUI_UNIICON "${APP_ICON_ICO}")                    # uninstaller
            set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\${PROJECT_NAME}.exe") # "Apps & features" entry
        endif()
    else()
        message(STATUS "NSIS not found -> building ZIP only (set -DMAKENSIS_EXECUTABLE=<path to makensis.exe>)")
    endif()

    include(CPack)

    # Run CPack at the end of "cmake --install".
    # CPack itself runs the install rules into its own staging folder, so NBMX_IN_CPACK
    # stops it from calling CPack again (endless loop).
    option(PACKAGE_ON_INSTALL "Run CPack at the end of cmake --install" ON)
    if(PACKAGE_ON_INSTALL)
        install(CODE "
            if(NOT DEFINED ENV{NBMX_IN_CPACK})
                message(STATUS \"Running CPack (\${CMAKE_INSTALL_CONFIG_NAME}) ...\")
                execute_process(
                    COMMAND \"${CMAKE_COMMAND}\" -E env NBMX_IN_CPACK=1
                            \"${CMAKE_CPACK_COMMAND}\"
                            --config \"${CMAKE_BINARY_DIR}/CPackConfig.cmake\"
                            -C \"\${CMAKE_INSTALL_CONFIG_NAME}\"
                    WORKING_DIRECTORY \"${CMAKE_BINARY_DIR}\"
                    COMMAND_ERROR_IS_FATAL ANY)
                # CPack's temporary staging copy (a full duplicate of the install) is not needed
                file(REMOVE_RECURSE \"${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages\")
            endif()
        ")
    endif()
endmacro()
