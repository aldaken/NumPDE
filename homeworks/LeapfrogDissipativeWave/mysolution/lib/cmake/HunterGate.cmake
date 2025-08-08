# Copyright (c) 2013-2019, Ruslan Baratov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This is a gate file to Hunter package manager.
# Include this file using `include` command and add package you need, example:
#
#     cmake_minimum_required(VERSION 3.2)
#
#     include("cmake/HunterGate.cmake")
#     HunterGate(
#         URL "https://github.com/path/to/hunter/archive.tar.gz"
#         SHA1 "798501e983f14b28b10cda16afa4de69eee1da1d"
#     )
#
#     project(MyProject)
#
#     hunter_add_package(Foo)
#     hunter_add_package(Boo COMPONENTS Bar Baz)
#
# Projects:
#     * https://github.com/hunter-packages/gate/
#     * https://github.com/ruslo/hunter

option(HUNTER_ENABLED "Enable Hunter package manager support" ON)

if(HUNTER_ENABLED)
  if(CMAKE_VERSION VERSION_LESS "3.2")
    message(
        FATAL_ERROR
        "At least CMake version 3.2 required for Hunter dependency management."
        " Update CMake or set HUNTER_ENABLED to OFF."
    )
  endif()
endif()

include(CMakeParseArguments) # cmake_parse_arguments

option(HUNTER_STATUS_PRINT "Print working status" ON)
option(HUNTER_STATUS_DEBUG "Print a lot info" OFF)
option(HUNTER_TLS_VERIFY "Enable/disable TLS certificate checking on downloads" ON)
set(HUNTER_ROOT "" CACHE FILEPATH "Override the HUNTER_ROOT.")

set(HUNTER_ERROR_PAGE "https://hunter.readthedocs.io/en/latest/reference/errors")

function(hunter_gate_status_print)
  if(HUNTER_STATUS_PRINT OR HUNTER_STATUS_DEBUG)
    foreach(print_message ${ARGV})
      message(STATUS "[hunter] ${print_message}")
    endforeach()
  endif()
endfunction()

function(hunter_gate_status_debug)
  if(HUNTER_STATUS_DEBUG)
    foreach(print_message ${ARGV})
      string(TIMESTAMP timestamp)
      message(STATUS "[hunter *** DEBUG *** ${timestamp}] ${print_message}")
    endforeach()
  endif()
endfunction()

function(hunter_gate_error_page error_page)
  message("------------------------------ ERROR ------------------------------")
  message("    ${HUNTER_ERROR_PAGE}/${error_page}.html")
  message("-------------------------------------------------------------------")
  message("")
  message(FATAL_ERROR "")
endfunction()

function(hunter_gate_internal_error)
  message("")
  foreach(print_message ${ARGV})
    message("[hunter ** INTERNAL **] ${print_message}")
  endforeach()
  message("[hunter ** INTERNAL **] [Directory:${CMAKE_CURRENT_LIST_DIR}]")
  message("")
  hunter_gate_error_page("error.internal")
endfunction()

function(hunter_gate_fatal_error)
  cmake_parse_arguments(hunter "" "ERROR_PAGE" "" "${ARGV}")
  if("${hunter_ERROR_PAGE}" STREQUAL "")
    hunter_gate_internal_error("Expected ERROR_PAGE")
  endif()
  message("")
  foreach(x ${hunter_UNPARSED_ARGUMENTS})
    message("[hunter ** FATAL ERROR **] ${x}")
  endforeach()
  message("[hunter ** FATAL ERROR **] [Directory:${CMAKE_CURRENT_LIST_DIR}]")
  message("")
  hunter_gate_error_page("${hunter_ERROR_PAGE}")
endfunction()

function(hunter_gate_user_error)
  hunter_gate_fatal_error(${ARGV} ERROR_PAGE "error.incorrect.input.data")
endfunction()

function(hunter_gate_self root version sha1 result)
  string(COMPARE EQUAL "${root}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("root is empty")
  endif()

  string(COMPARE EQUAL "${version}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("version is empty")
  endif()

  string(COMPARE EQUAL "${sha1}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("sha1 is empty")
  endif()

  string(SUBSTRING "${sha1}" 0 7 archive_id)

  if(EXISTS "${root}/cmake/Hunter")
    set(hunter_self "${root}")
  else()
    set(
        hunter_self
        "${root}/_Base/Download/Hunter/${version}/${archive_id}/Unpacked"
    )
  endif()

  set("${result}" "${hunter_self}" PARENT_SCOPE)
endfunction()

# Set HUNTER_GATE_ROOT cmake variable to suitable value.
function(hunter_gate_detect_root)
  # Check CMake variable
  if(HUNTER_ROOT)
    set(HUNTER_GATE_ROOT "${HUNTER_ROOT}" PARENT_SCOPE)
    hunter_gate_status_debug("HUNTER_ROOT detected by cmake variable")
    return()
  endif()

  # Check environment variable
  if(DEFINED ENV{HUNTER_ROOT})
    set(HUNTER_GATE_ROOT "$ENV{HUNTER_ROOT}" PARENT_SCOPE)
    hunter_gate_status_debug("HUNTER_ROOT detected by environment variable")
    return()
  endif()

  # Check HOME environment variable
  if(DEFINED ENV{HOME})
    set(HUNTER_GATE_ROOT "$ENV{HOME}/.hunter" PARENT_SCOPE)
    hunter_gate_status_debug("HUNTER_ROOT set using HOME environment variable")
    return()
  endif()

  # Check SYSTEMDRIVE and USERPROFILE environment variable (windows only)
  if(WIN32)
    if(DEFINED ENV{SYSTEMDRIVE})
      set(HUNTER_GATE_ROOT "$ENV{SYSTEMDRIVE}/.hunter" PARENT_SCOPE)
      hunter_gate_status_debug(
          "HUNTER_ROOT set using SYSTEMDRIVE environment variable"
      )
      return()
    endif()

    if(DEFINED ENV{USERPROFILE})
      set(HUNTER_GATE_ROOT "$ENV{USERPROFILE}/.hunter" PARENT_SCOPE)
      hunter_gate_status_debug(
          "HUNTER_ROOT set using USERPROFILE environment variable"
      )
      return()
    endif()
  endif()

  hunter_gate_fatal_error(
      "Can't detect HUNTER_ROOT"
      ERROR_PAGE "error.detect.hunter.root"
  )
endfunction()

function(hunter_gate_download dir)
  string(
      COMPARE
      NOTEQUAL
      "$ENV{HUNTER_DISABLE_AUTOINSTALL}"
      ""
      disable_autoinstall
  )
  if(disable_autoinstall AND NOT HUNTER_RUN_INSTALL)
    hunter_gate_fatal_error(
        "Hunter not found in '${dir}'"
        "Set HUNTER_RUN_INSTALL=ON to auto-install it from '${HUNTER_GATE_URL}'"
        "Settings:"
        "  HUNTER_ROOT: ${HUNTER_GATE_ROOT}"
        "  HUNTER_SHA1: ${HUNTER_GATE_SHA1}"
        ERROR_PAGE "error.run.install"
    )
  endif()
  string(COMPARE EQUAL "${dir}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("Empty 'dir' argument")
  endif()

  string(COMPARE EQUAL "${HUNTER_GATE_SHA1}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("HUNTER_GATE_SHA1 empty")
  endif()

  string(COMPARE EQUAL "${HUNTER_GATE_URL}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("HUNTER_GATE_URL empty")
  endif()

  set(done_location "${dir}/DONE")
  set(sha1_location "${dir}/SHA1")

  set(build_dir "${dir}/Build")
  set(cmakelists "${dir}/CMakeLists.txt")

  hunter_gate_status_debug("Locking directory: ${dir}")
  file(LOCK "${dir}" DIRECTORY GUARD FUNCTION)
  hunter_gate_status_debug("Lock done")

  if(EXISTS "${done_location}")
    # while waiting for lock other instance can do all the job
    hunter_gate_status_debug("File '${done_location}' found, skip install")
    return()
  endif()

  file(REMOVE_RECURSE "${build_dir}")
  file(REMOVE_RECURSE "${cmakelists}")

  file(MAKE_DIRECTORY "${build_dir}") # check directory permissions

  # Disabling languages speeds up a little bit, reduces noise in the output
  # and avoids path too long windows error
  file(
      WRITE
      "${cmakelists}"
      "cmake_minimum_required(VERSION 3.2)\n"
      "project(HunterDownload LANGUAGES NONE)\n"
      "include(ExternalProject)\n"
      "ExternalProject_Add(\n"
      "    Hunter\n"
      "    URL\n"
      "    \"${HUNTER_GATE_URL}\"\n"
      "    URL_HASH\n"
      "    SHA1=${HUNTER_GATE_SHA1}\n"
      "    DOWNLOAD_DIR\n"
      "    \"${dir}\"\n"
      "    TLS_VERIFY\n"
      "    ${HUNTER_TLS_VERIFY}\n"
      "    SOURCE_DIR\n"
      "    \"${dir}/Unpacked\"\n"
      "    CONFIGURE_COMMAND\n"
      "    \"\"\n"
      "    BUILD_COMMAND\n"
      "    \"\"\n"
      "    INSTALL_COMMAND\n"
      "    \"\"\n"
      ")\n"
  )

  if(HUNTER_STATUS_DEBUG)
    set(logging_params "")
  else()
    set(logging_params OUTPUT_QUIET)
  endif()

  hunter_gate_status_debug("Run generate")

  # Need to add toolchain file too.
  # Otherwise on Visual Studio + MDD this will fail with error:
  # "Could not find an appropriate version of the Windows 10 SDK installed on this machine"
  if(EXISTS "${CMAKE_TOOLCHAIN_FILE}")
    get_filename_component(absolute_CMAKE_TOOLCHAIN_FILE "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE)
    set(toolchain_arg "-DCMAKE_TOOLCHAIN_FILE=${absolute_CMAKE_TOOLCHAIN_FILE}")
  else()
