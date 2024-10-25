ifeq ($(shell test -e /bin/cygwin1.dll && echo found),found)
PROG	= dmsd.exe
else
PROG	= dmsd
endif

DESTLIB	= ${LIBDIR}/dms

DEFS	= -DDESTLIB=\"${DESTLIB}\"

OBJS	= server.o viewat.o lookat.o freadv.o emalloc.o
OPT	= -g
CFLAGS	= -I.. ${DEFS} ${OPT}

${PROG}: ${OBJS}
	${CC} -o ${PROG} ${OBJS} -lm

beforeinstall:
	-mkdir ${DESTLIB}

afterinstall:
	cp ../dms_servers.proto ${DESTLIB}/dms_servers.proto

install: ${PROG} afterinstall
afterinstall: realinstall
realinstall: beforeinstall
	cp ${PROG} ${DESTLIB}
	-strip ${DESTLIB}/${PROG}

clean:
	rm -f ${OBJS}

spotless:
	rm -f ${OBJS} ${PROG}
