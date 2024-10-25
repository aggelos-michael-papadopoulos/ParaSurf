/*
 *	Copyright (c) 1994 The Regents of the University of California.
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
 *	$Id: pdbrun.c,v 1.4 1994/11/18 01:00:57 gregc Exp $
 *
 *	subroutine for reading PDB format files
 *
 */

/* LINTLIBRARY */

# include	"pdb_int.h"
# include	<string.h>

int	pdb_pdbrun_version = PDB_PDBRUN_VERSION;

#ifdef __STDC__
const char *
pdb_gfx_string(int i)
#else
char *
pdb_gfx_string(i)
	int	i;
#endif
{
	switch (i) {
	default:			return "UNKNOWN";
	case PDB_GFX_POINTS:		return "POINTS";
	case PDB_GFX_MARKERS:		return "MARKERS";
	case PDB_GFX_LINES:		return "LINES";
	case PDB_GFX_LINE_STRIP:	return "LINE-STRIP";
	case PDB_GFX_LINE_LOOP:		return "LINE-LOOP";
	case PDB_GFX_TRIANGLES:		return "TRIANGLES";
	case PDB_GFX_TRIANGLE_STRIP:	return "TRIANGLE-STRIP";
	case PDB_GFX_TRIANGLE_FAN:	return "TRIANGLE-FAN";
	case PDB_GFX_QUADS:		return "QUADS";
	case PDB_GFX_QUAD_STRIP:	return "QUAD-STRIP";
	case PDB_GFX_POLYGON:		return "POLYGON";
	}
}

#ifdef __STDC__
int
pdb_gfx_type(const char *type)
#else
int
pdb_gfx_type(type)
	char	*type;
#endif
{
	switch (type[0]) {
	case 'L':
		if (strcmp(type + 1, "INE-LOOP") == 0)
			return PDB_GFX_LINE_LOOP;
		if (strcmp(type + 1, "INE-STRIP") == 0)
			return PDB_GFX_LINE_STRIP;
		if (strcmp(type + 1, "INES") == 0)
			return PDB_GFX_LINES;
		break;
	case 'M':
		if (strcmp(type + 1, "ARKERS") == 0)
			return PDB_GFX_MARKERS;
		break;
	case 'P':
		if (strcmp(type + 1, "OINTS") == 0)
			return PDB_GFX_POINTS;
		if (strcmp(type + 1, "OLYGON") == 0)
			return PDB_GFX_POLYGON;
		break;
	case 'Q':
		if (strcmp(type + 1, "UAD-STRIP") == 0)
			return PDB_GFX_QUAD_STRIP;
		if (strcmp(type + 1, "UADS") == 0)
			return PDB_GFX_QUADS;
		break;
	case 'T':
		if (strcmp(type + 1, "RIANGLE-FAN") == 0)
			return PDB_GFX_TRIANGLE_FAN;
		if (strcmp(type + 1, "RIANGLE-STRIP") == 0)
			return PDB_GFX_TRIANGLE_STRIP;
		if (strcmp(type + 1, "RIANGLES") == 0)
			return PDB_GFX_TRIANGLES;
		break;
	}
	return PDB_GFX_UNKNOWN;
}
