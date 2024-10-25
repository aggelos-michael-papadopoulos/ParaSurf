# LIBDIR is the directory where a 'dms' subdirectory will be created,
# in which the server binary (dmsd) and various support files will be installed
LIBDIR  = /usr/local/lib

# BINDIR is the directory where the dms binary will be installed
BINDIR  = /usr/local/bin

# PDBINC is the directory where the 'pdb.h' file from the UCSF PDB-reading
# library got installed.  'pdb.h' is not installed by default by that
# distribution, so you will have to install it by hand into PDBINC
PDBINC  = libpdb

# PDBLIB is the directory where libpdb.a from the UCSF PDB-reading library 
# got installed.
PDBLIB  = libpdb

# MANDIR is the directory where the dms.1 manual page will be installed
MANDIR  = /usr/local/man/man1

ifeq ($(shell test -e /bin/cygwin1.dll && echo found),found)
	PROG	= dms.exe
else
	PROG	= dms
endif

# you should not need to modify any of the following
DESTLIB	= ${LIBDIR}/dms/
SERVER	= ${DESTLIB}dmsd
DEFS	= -DDESTLIB=\"${DESTLIB}\" -DSERVER_PATH=\"$(SERVER)\"
OPT	= -O
CFLAGS  = ${DEFS} -I${PDBINC} ${OPT}

OBJS	= compute.o fwritev.o input.o ms.o output.o emalloc.o tokenize.o

all:	${PROG}
	cd dmsd ; ${MAKE} LIBDIR=${LIBDIR} OPT="${OPT}"

${PROG}: pdb ${OBJS}
	${CC} -L${PDBLIB} -o ${PROG} ${OBJS} -lpdb

pdb:
	cd libpdb ; ${MAKE} OPT="${OPT}"

install: ${PROG} maninstall afterinstall
afterinstall: realinstall
realinstall: beforeinstall
	cp ${PROG} ${BINDIR}
	-strip ${BINDIR}/${PROG}

maninstall:
	-rm -f ${MANDIR}/dms.1
	cp dms.1 ${MANDIR}

beforeinstall:
	if [ -d ${DESTLIB} ]; then true ; else mkdir ${DESTLIB}; fi
	cd dmsd ; ${MAKE} LIBDIR=${LIBDIR} install

afterinstall:
	cp radii.proto ${DESTLIB}radii

clean:
	/bin/rm -f ${OBJS}
	cd libpdb ; ${MAKE} clean
	cd dmsd ; ${MAKE} clean

spotless:
	/bin/rm -f ${OBJS} ${PROG}
	cd libpdb ; ${MAKE} spotless
	cd dmsd ; ${MAKE} spotless
