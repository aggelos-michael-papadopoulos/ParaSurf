SHELL		= /bin/sh
ifeq ($(shell test -e /usr/bin/ranlib && echo found),found)
	RANLIB		= ranlib
else
	RANLIB		= :
endif

OPT		= -g
DEBUG		=
CFLAGS		= $(OPT) $(DEBUG)
LIBRARY		= pdb

LIBDIR		= /usr/local/lib
LIBARCH		= lib$(LIBRARY).a
OBJS		= pdb_read.o pdb_sprntf.o pdb_sscanf.o pdb_write.o pdbrun.o ms.o
SRCS		= pdb_read.c pdb_sprntf.c pdb_sscanf.c pdb_write.c pdbrun.c ms.c

all:		$(LIBARCH)

$(LIBARCH):     $(OBJS)
		ar cru $(LIBARCH) $(OBJS)
		$(RANLIB) $(LIBARCH)

install:	$(LIBARCH)
		cp $(LIBARCH) $(LIBDIR)
		$(RANLIB) $(LIBDIR)/$(LIBARCH)

clean:
		rm -f $(OBJS)

spotless:
		rm -f $(OBJS) $(LIBARCH)

tags:           $(HDRS) $(SRCS); @ctags -w $(HDRS) $(SRCS)
