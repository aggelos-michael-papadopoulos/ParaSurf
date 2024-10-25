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
 *	$Id: ms.c,v 2.4 94/05/19 16:01:16 conrad Exp $
 */

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "pdb_int.h"
#include "ms.h"

#ifndef TRUE
#define	TRUE	1
#define	FALSE	0
#endif

/*
 * ms_read_record:
 *	Read an ms record from a file and return the corresponding
 *	data structure
 */
ms_record
#ifdef __STDC__
ms_read_record(FILE *fp)
#else
ms_read_record(fp)
	FILE	*fp;
#endif
{
	char		buf[MS_BUFSIZ];
	char		*cp;
	int		c;
	ms_record	record;

	/*
	 * If at end of file, just set the record type and return
	 */
	if (fgets(buf, sizeof buf, fp) == NULL) {
		record.record_type = MS_END;
		return record;
	}

	cp = strchr(buf, '\n');
	if (cp != NULL)
		*cp = '\0';
	else
		/* discard extra characters since line too long */
		while ((c = getc(fp)) != '\n' && c != EOF)
			continue;

	return ms_read_string(buf);
}

#ifdef __STDC__
	static void	crunch(char *);
#else
	static void	crunch();
#endif

/*
 * ms_read_string
 *	Construct the data structure corresponding to the given line
 */
ms_record
#ifdef __STDC__
ms_read_string(const char *buf)
#else
ms_read_string(buf)
	char	*buf;
#endif
{
	int		i;
	const char	*cp;
	char		*sp;
	int		nfield;
	double		area, extra;
	double		coord[3], normal[3];
	ms_record	record;
	ms_atom		*map;
	ms_surface	*msp;
	char		type;
	const char	*atom_fmt, *surf_fmt;

	/*
	 * If the head of the string is "user" (case-independently),
	 * we copy the buffer into the string and return
	 */
	if (strncasecmp(buf, "user", 4) == 0) {
		record.record_type = MS_USER;
		for (cp = buf + 4; isspace(*cp); cp++)
			continue;
		sp = record.ms.user.string;
		while (*cp != '\0' && *cp != '\n')
			*sp++ = *cp++;
		*sp = '\0';
		return record;
	}

	/*
	 * This is either an atom line, surface line, or an error
	 */
	if (strlen(buf) < 41) {
		record.record_type = MS_UNKNOWN;
		(void) strcpy(record.ms.unknown.string, buf);
		return record;
	}
	type = buf[40];
	atom_fmt = "%3s %4s %4s%8f %8f %8f A";
	surf_fmt = "%3s %4s %4s%8f %8f %8f S%c%c %6f %6f %6f %6f %6f";
	if (isspace(type)) {
		type = buf[41];
		atom_fmt = "%3s %5s %4s%8f %8f %8f A";
		surf_fmt = "%3s %5s %4s%8f %8f %8f S%c%c %6f %6f %6f %6f %6f";
	}
	switch (type) {
	  case 'A':
		record.record_type = MS_ATOM;
		map = &record.ms.atom;
		nfield = pdb_sscanf(buf, atom_fmt,
			map->residue_type, map->residue_sequence,
			map->atom_name, &coord[0], &coord[1], &coord[2]);
		if (nfield == 6) {
			for (i = 0; i < 3; i++)
				map->coord[i] = coord[i];
			crunch(map->residue_type);
			crunch(map->residue_sequence);
			crunch(map->atom_name);
			return record;
		}
		break;
	  case 'S':
		record.record_type = MS_SURFACE;
		msp = &record.ms.surface;
		nfield = pdb_sscanf(buf, surf_fmt,
			msp->residue_type, msp->residue_sequence,
			msp->atom_name, &coord[0], &coord[1], &coord[2],
			&msp->type, &msp->level, &area,
			&normal[0], &normal[1], &normal[2], &extra);
		switch (nfield) {
		  case 9:
			normal[0] = 0;
			/* FALLTHROUGH */
		  case 10:
			msp->has_normal = FALSE;
			for (i = 0; i < 3; i++)
				msp->coord[i] = coord[i];
			msp->area = area;
			msp->extra = normal[0];
			crunch(msp->residue_type);
			crunch(msp->residue_sequence);
			crunch(msp->atom_name);
			return record;
		  case 12:
			extra = 0;
			/* FALLTHROUGH */
		  case 13:
			msp->has_normal = TRUE;
			for (i = 0; i < 3; i++) {
				msp->coord[i] = coord[i];
				msp->normal[i] = normal[i];
			}
			msp->area = area;
			msp->extra = extra;
			crunch(msp->residue_type);
			crunch(msp->residue_sequence);
			crunch(msp->atom_name);
			return record;
		}
		break;
	}

	/*
	 * Okay.  We must not have recognized the record.  Set the
	 * record type, copy the string into the string and return
	 */
	record.record_type = MS_UNKNOWN;
	(void) strcpy(record.ms.unknown.string, buf);
	return record;
}

/*
 * crunch:
 *	Remove blanks from a string
 */
static
void
#ifdef __STDC__
crunch(char *s)
#else
crunch(s)
	char	*s;
#endif
{
	char	*last;

	for (last = s; *s != '\0'; s++)
		if (!isspace(*s))
			*last++ = *s;
	*last = '\0';
}

/*
 * ms_write_record:
 *	Write an ms record to a file from the corresponding
 *	data structure
 */
void
#ifdef __STDC__
ms_write_record(FILE *fp, const ms_record *mp)
#else
ms_write_record(fp, mp)
	FILE		*fp;
	ms_record	*mp;
#endif
{
	char	buffer[MS_RECLEN];

	ms_write_string(buffer, mp);
	fprintf(fp, "%s\n", buffer);
}

/*
 * ms_write_string:
 *	Write an ms record to a string from the corresponding
 *	data structure
 */
void
#ifdef __STDC__
ms_write_string(char *buffer, const ms_record *mp)
#else
ms_write_string(buffer, mp)
	char		*buffer;
	ms_record	*mp;
#endif
{
	const ms_atom		*map;
	const ms_surface	*msp;
	char			*cp;

	switch (mp->record_type) {
	  case MS_ATOM:
		map = &mp->ms.atom;
		(void) sprintf(buffer, "%-3s %4s %4s %7.3f  %7.3f  %7.3f A",
			map->residue_type, map->residue_sequence,
			map->atom_name, map->coord[0], map->coord[1],
			map->coord[2]);
		break;
	  case MS_SURFACE:
		msp = &mp->ms.surface;
		(void) sprintf(buffer,
			"%-3s %4s %4s %7.3f  %7.3f  %7.3f S%c%c %6.3f",
			msp->residue_type, msp->residue_sequence,
			msp->atom_name, msp->coord[0], msp->coord[1],
			msp->coord[2], msp->type, msp->level, msp->area);
		cp = strchr(buffer, '\0');
		if (msp->has_normal) {
			(void) sprintf(cp, " %6.3f %6.3f %6.3f",
				msp->normal[0], msp->normal[1], msp->normal[2]);
			cp = strchr(cp, '\0');
		}
		if (msp->extra != 0) {
			if (msp->extra >= 1000 || msp->extra <= -100)
				(void) sprintf(cp, " %6.1f", msp->extra);
			else if (msp->extra >= 100 || msp->extra <= -10)
				(void) sprintf(cp, " %6.2f", msp->extra);
			else
				(void) sprintf(cp, " %6.3f", msp->extra);
		}
		break;
	  case MS_USER:
		(void) sprintf(buffer, "USER %s\n", mp->ms.user.string);
		break;
	}
}
