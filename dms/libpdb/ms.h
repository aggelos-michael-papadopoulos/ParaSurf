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
 *	$Id: ms.h,v 2.2 94/01/27 12:41:26 gregc Exp $
 */

#ifndef MS_H
#define	MS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define	MS_RECLEN	128
#define	MS_BUFSIZ	(MS_RECLEN + 2)

#define	MS_MAX_RESTYPE	8
#define	MS_MAX_RESSEQ	8
#define	MS_MAX_ATNAME	8

#define	MS_UNKNOWN	0
#define	MS_END		1
#define	MS_ATOM		2
#define	MS_SURFACE	3
#define	MS_USER		4

#define	MS_NUM_R	5

typedef struct ms_atom	{
	char	residue_type[MS_MAX_RESTYPE];
	char	residue_sequence[MS_MAX_RESSEQ];
	char	atom_name[MS_MAX_ATNAME];
	float	coord[3];
}	ms_atom;

typedef struct ms_surf	{
	char	residue_type[MS_MAX_RESTYPE];
	char	residue_sequence[MS_MAX_RESSEQ];
	char	atom_name[MS_MAX_ATNAME];
	float	coord[3];
	float	normal[3];
	float	area;
	float	extra;
	char	has_normal;
	char	type;
	char	level;
}	ms_surface;

typedef struct ms_user	{
	char	string[MS_BUFSIZ];
}	ms_user;

#define	ms_unknown	ms_user

typedef struct ms_record	{
	int	record_type;
	union	{
		ms_unknown	unknown;
		ms_atom		atom;
		ms_surface	surface;
		ms_user		user;
	}	ms;
}	ms_record;

#if defined(__STDC__) || defined(__cplusplus)
extern ms_record	ms_read_record(FILE *);
extern ms_record	ms_read_string(const char *);
extern void		ms_write_record(FILE *, const ms_record *);
extern void		ms_write_string(char *, const ms_record *);
#else
extern ms_record	ms_read_record();
extern ms_record	ms_read_string();
extern void		ms_write_record();
extern void		ms_write_string();
#endif

#ifdef __cplusplus
}
#endif
#endif
