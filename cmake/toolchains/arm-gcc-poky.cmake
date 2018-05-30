set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ARM64)
set(SNN_POKY_ROOT $ENV{SNN_POKY_ROOT})

if(NOT SNN_POKY_ROOT)
  message(FATAL_ERROR
    "Please set SNN_POKY_ROOT in the environment when crosscompiling.")
endif()

set(SNN_TARGET_TRIPLE aarch64-poky-linux)
set(SNN_TOOLCHAIN_DIR ${SNN_POKY_ROOT}/x86_64-pokysdk-linux)
set(SNN_SYSROOT_DIR ${SNN_POKY_ROOT}/aarch64-poky-linux)
# Adding this as the GCC toolchain makes compute++ not find headers
set(SNN_DONT_USE_TOOLCHAIN ON)

set(CMAKE_C_COMPILER "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-gcc" CACHE PATH "gcc")
set(CMAKE_CXX_COMPILER "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-g++" CACHE PATH "g++")
set(CMAKE_AR "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-ar" CACHE PATH "archive")
set(CMAKE_LINKER "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-ld" CACHE PATH "linker")
set(CMAKE_NM "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-nm" CACHE PATH "nm")
set(CMAKE_OBJCOPY "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-objcopy" CACHE PATH "objcopy")
set(CMAKE_OBJDUMP "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-objdump" CACHE PATH "objdump")
set(CMAKE_STRIP "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-strip" CACHE PATH "strip")
set(CMAKE_RANLIB "${SNN_TOOLCHAIN_DIR}/usr/bin/${SNN_TARGET_TRIPLE}/${SNN_TARGET_TRIPLE}-ranlib" CACHE PATH "ranlib")

set(CMAKE_FIND_ROOT_PATH ${SNN_SYSROOT_DIR})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

set(CMAKE_SYSROOT "${SNN_SYSROOT_DIR}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__aarch64__ --sysroot=${SNN_SYSROOT_DIR}" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__aarch64__ --sysroot=${SNN_SYSROOT_DIR}" CACHE INTERNAL "")

set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>" CACHE INTERNAL "")
