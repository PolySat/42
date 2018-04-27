include Make.rules.arm
BUILD_DIR?=$(shell pwd)
DIST_FILE:=$(BUILD_DIR)/dist/nasa42-1.0.linux-x86_64.tar
SLEN=$(shell dirname `tar -tf $(DIST_FILE) | grep packages | head -1` | sed -e 's/[^/]//g' | wc -c)

GUIFLAG = -D _USE_GUI_
#GUIFLAG = 

SHADERFLAG = -D _USE_SHADERS_
#SHADERFLAG = 

TIMEFLAG = 
#TIMEFLAG = -D _USE_SYSTEM_TIME_

CFDFLAG = 
#CFDFLAG = -D _ENABLE_CFD_SLOSH_

FFTBFLAG = 
#FFTBFLAG = -D _ENABLE_FFTB_CODE_

#GSFCFLAG = 
GSFCFLAG = -D _USE_GSFC_WATERMARK_

# Linux Macros
CINC =
EXTERNDIR =  

#   SOCKETFLAG = 
SOCKETFLAG = -D _ENABLE_SOCKETS_

ifneq ($(strip $(GUIFLAG)),)
	GUIOBJ = $(OBJ)42GlutGui.o $(OBJ)glkit.o 
	#GLINC = -I /usr/include/
	GLINC = -I $(KITDIR)/include/GL/
	LIBS = -lglut -lGLU -lGL 
	LFLAGS = -L $(KITDIR)/GL/lib/ -shared -fPIC -Wl,-soname,$(SO_NAME)
	ARCHFLAG = 
else
	GUIOBJ = 
	GLINC = 
	LIBS = 
	LFLAGS = 
	ARCHFLAG = 
endif
EXENAME = 42
CC = gcc

# Basic directories
HOMEDIR = ../
PROJDIR = ../
KITDIR = $(PROJDIR)Kit/
OBJ = $(PROJDIR)Object/
INC = $(PROJDIR)Include/
SRC = $(PROJDIR)Source/
KITINC = $(KITDIR)Include/
KITSRC = $(KITDIR)Source/
INOUT = $(PROJDIR)InOut/
GSFCSRC = $(PROJDIR)/GSFC/Source/

# If not using IPC, don't compile IPC-related files
ifneq ($(strip $(SOCKETFLAG)),)
   IPCOBJ = $(OBJ)42ipc.o
else
   IPCOBJ = 
endif

# If not in FFTB, don't compile FFTB-related files
ifneq ($(strip $(FFTBFLAG)),)
   FFTBOBJ = $(OBJ)42fftb.o
else
   FFTBOBJ = 
endif

ANSIFLAGS = -Werror
CFLAGS = -Wall -Wshadow -Wno-deprecated  $(ANSIFLAGS) $(GLINC) $(CINC) -I $(INC) \
 -I $(KITINC) -I $(KITSRC) -I $(MATLABSRC) -I $(MATLABINC) -I $(SIMULINKINC) -O0 \
 $(ARCHFLAG) $(GUIFLAG) $(SHADERFLAG) $(TIMEFLAG) $(SOCKETFLAG) $(EMBEDDED) $(CFDFLAG) \
  $(FFTBFLAG) $(GSFCFLAG) -DNPY_NO_DEPRECATED_API="NPY_1_7_API_VERSION"

all:
	CC="$(CC)" CFLAGS="$(CFLAGS) -L../" LDSHARED="$(CC) -shared -Wl,-E" python setup.py bdist --format=tar

TEST=$(shell dirname `tar -tf $(DIST_FILE) | grep packages | head -1` | sed -e 's/[^/]//g')

install:
	tar -C $(TARGET_PATH) --strip-components=$(SLEN) -xf $(DIST_FILE)

clean:
	rm -fr *.o build
