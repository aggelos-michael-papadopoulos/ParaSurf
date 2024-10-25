/*
 *	Copyright (c) 1989 The Regents of the University of California.
 *	All rights reserved.
 *
 *	Redistribution and use in source and binary forms are permitted
 *	provided that the above copyright notice and this paragraph are
 *	duplicated in all such forms and that any documentation,
 *	advertising materials, and other materials related to such
 *	distribution and use acknowledge that the software was developed
 *	by the University of California, San Francisco.  The name of the
 *	University may not be used to endorse or promote products derived
 *	from this software without specific prior written permission.
 *	THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 *	IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 *	WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 *	$Id: pdb_int.h,v 2.4 1994/11/17 00:58:58 gregc Exp $
 */

#include	"pdb.h"

#ifndef __STDC__
# define	const
#endif

extern int	pdb_pdbrun_version;

#ifdef __STDC__
extern int	pdb_sscanf(const char *, const char *, ...);
extern void	pdb_sprintf(char *, const char *, ...);
extern const char	*pdb_gfx_string(int i);
extern int	pdb_gfx_type(const char *type);
#else
extern int	pdb_sscanf();
extern void	pdb_sprintf();
extern char	*pdb_gfx_string();
extern int	pdb_gfx_type();
#endif
