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
 *	$Id: pdb_read.c,v 2.20 1995/07/22 02:03:18 gregc Exp $
 *
 *	subroutine for reading PDB format files
 *
 */

/* LINTLIBRARY */

# include	<stdio.h>
# include	<ctype.h>
# include	<string.h>
# include	"pdb_int.h"

# ifndef _tolower
# define	_tolower	tolower
# endif

# define	STREQN(a,b,n)	(strncmp(a, b, n) == 0)

static const char * const pdb_record_format[PDB_NUM_R] = {
#include "read_format.i"
};

static char const * const pdbrun5[] = {
#include "pdbrun5_read.i"
};

static char const * const pdbrun6[] = {
#include "pdbrun6_read.i"
};

/*
 *	for each pdb record type there is a format reading in the
 *	record values and for printing them out.
 *
 *	The actual format of a line written, is the print format
 *	followed by blank padding to 72 characters, followed by
 *	8 characters of file and line information.
 */

pdb_record
# ifdef __STDC__
pdb_read_record(FILE *f)
# else
pdb_read_record(f)
	FILE		*f;
# endif
{

	char			buffer[PDB_BUFSIZ];
	char			*cp;
	int			c;
	static pdb_record	r_end = { PDB_END };

	if (fgets(buffer, PDB_BUFSIZ, f) == NULL) {
		/* at eof or error - default to eof */
		return r_end;
	}

	cp = strchr(buffer, '\n');
	if (cp != NULL)
		*cp = '\0';
	else
		/* discard extra characters since line too long */
		while ((c = getc(f)) != '\n' && c != EOF)
			continue;

	return pdb_read_string(buffer);
}

static int
#ifdef __STDC__
pdbrun5_type(const char *buf)
#else
pdbrun5_type(buf)
	char	*buf;
#endif
{
	switch (buf[0]) {
	case 'A': case 'a':
		if (strncasecmp(buf + 1, "NGLE ", 5) == 0)
			return PDB_USER_ANGLE;
		if (strncasecmp(buf + 1, "TPOS ", 5) == 0)
			return PDB_USER_ATPOS;
		break;
	case 'B': case 'b':
		if (strncasecmp(buf + 1, "GCOLOR ", 7) == 0)
			return PDB_USER_BGCOLOR;
		break;
	case 'C': case 'c':
		if (strncasecmp(buf + 1, "HAIN ", 5) == 0)
			return PDB_USER_CHAIN;
		if (strncasecmp(buf + 1, "NAME ", 5) == 0)
			return PDB_USER_CNAME;
		if (strncasecmp(buf + 1, "OLOR ", 5) == 0)
			return PDB_USER_COLOR;
		break;
	case 'D': case 'd':
		if (strncasecmp(buf + 1, "ISTANCE ", 8) == 0)
			return PDB_USER_DISTANCE;
		break;
	case 'E': case 'e':
		if (strncasecmp(buf + 1, "NDOBJ ", 6) == 0)
			return PDB_USER_ENDOBJ;
		if (strncasecmp(buf + 1, "YEPOS ", 6) == 0)
			return PDB_USER_EYEPOS;
		break;
	case 'F': case 'f':
		if (strncasecmp(buf + 1, "ILE ", 4) == 0)
			return PDB_USER_FILE;
		if (strncasecmp(buf + 1, "OCUS ", 5) == 0)
			return PDB_USER_FOCUS;
		break;
	case 'G': case 'g':
		if (strncasecmp(buf + 1, "FX ", 3) != 0)
			break;
		if (strncasecmp(buf + 4, "COLOR ", 6) == 0)
			return PDB_USER_GFX_COLOR;
		if (strncasecmp(buf + 4, "DRAW ", 5) == 0)
			return PDB_USER_GFX_DRAW;
		if (strncasecmp(buf + 4, "FONT ", 5) == 0)
			return PDB_USER_GFX_FONT;
		if (strncasecmp(buf + 4, "LABEL ", 6) == 0)
			return PDB_USER_GFX_LABEL;
		if (strncasecmp(buf + 4, "MARKER ", 7) == 0)
			return PDB_USER_GFX_MARKER;
		if (strncasecmp(buf + 4, "MOVE ", 5) == 0)
			return PDB_USER_GFX_MOVE;
		if (strncasecmp(buf + 4, "POINT ", 6) == 0)
			return PDB_USER_GFX_POINT;
		break;
	case 'O': case 'o':
		if (strncasecmp(buf + 1, "BJECT ", 6) == 0)
			return PDB_USER_OBJECT;
		break;
	case 'P': case 'p':
		if (strncasecmp(buf + 1, "DBRUN ", 6) == 0)
			return PDB_USER_PDBRUN;
		break;
	case 'R': case 'r':
		if (strncasecmp(buf + 1, "ADIUS ", 6) == 0)
			return PDB_USER_RADIUS;
		break;
	case 'V': case 'v':
		if (strncasecmp(buf + 1, "IEWPORT ", 8) == 0)
			return PDB_USER_VIEWPORT;
		break;
	case 'W': case 'w':
		if (strncasecmp(buf + 1, "INDOW ", 6) == 0)
			return PDB_USER_WINDOW;
		break;
	}
	return PDB_USER;
}

static int
#ifdef __STDC__
pdbrun6_type(const char *buf)
#else
pdbrun6_type(buf)
	char	*buf;
#endif
{
	switch (buf[0]) {
	case 'A': case 'a':
		if (strncasecmp(buf + 1, "NGLE ", 5) == 0)
			return PDB_USER_ANGLE;
		if (strncasecmp(buf + 1, "TPOS ", 5) == 0)
			return PDB_USER_ATPOS;
		break;
	case 'B': case 'b':
		if (strncasecmp(buf + 1, "GCOLOR ", 7) == 0)
			return PDB_USER_BGCOLOR;
		break;
	case 'C': case 'c':
		if (strncasecmp(buf + 1, "HAIN ", 5) == 0)
			return PDB_USER_CHAIN;
		if (strncasecmp(buf + 1, "NAME ", 5) == 0)
			return PDB_USER_CNAME;
		if (strncasecmp(buf + 1, "OLOR ", 5) == 0)
			return PDB_USER_COLOR;
		break;
	case 'D': case 'd':
		if (strncasecmp(buf + 1, "ISTANCE ", 8) == 0)
			return PDB_USER_DISTANCE;
		break;
	case 'E': case 'e':
		if (strncasecmp(buf + 1, "NDOBJ", 5) == 0
		&& (buf[6] == '\0' || buf[6] == '\n' || buf[6] == ' '))
			return PDB_USER_ENDOBJ;
		if (strncasecmp(buf + 1, "YEPOS ", 6) == 0)
			return PDB_USER_EYEPOS;
		break;
	case 'F': case 'f':
		if (strncasecmp(buf + 1, "ILE ", 4) == 0)
			return PDB_USER_FILE;
		if (strncasecmp(buf + 1, "OCUS ", 5) == 0)
			return PDB_USER_FOCUS;
		break;
	case 'G': case 'g':
		if (buf[1] != 'F' || buf[2] != 'X' || buf[3] != ' ')
			break;
		switch (buf[4]) {
		case 'B': case 'b':
			if (strncasecmp(buf + 5, "EGIN ", 5) == 0)
				return PDB_USER_GFX_BEGIN;
			break;
		case 'C': case 'c':
			if (strncasecmp(buf + 5, "OLOR ", 5) == 0)
				return PDB_USER_GFX_COLOR;
			break;
		case 'E': case 'e':
			if (buf[5] == 'N' && buf[6] == 'D'
			&& (buf[7] == '\0' || buf[7] == '\n' || buf[7] == ' '))
				return PDB_USER_GFX_END;
			break;
		case 'F': case 'f':
			if (strncasecmp(buf + 5, "ONT ", 4) == 0)
				return PDB_USER_GFX_FONT;
			break;
		case 'L': case 'l':
			if (strncasecmp(buf + 5, "ABEL ", 5) == 0)
				return PDB_USER_GFX_LABEL;
			break;
		case 'N': case 'n':
			if (strncasecmp(buf + 5, "ORMAL ", 6) == 0)
				return PDB_USER_GFX_NORMAL;
			break;
		case 'T': case 't':
			if (strncasecmp(buf + 5, "EXTPOS ", 7) == 0)
				return PDB_USER_GFX_TEXTPOS;
			break;
		case 'V': case 'v':
			if (strncasecmp(buf + 5, "ERTEX ", 6) == 0)
				return PDB_USER_GFX_VERTEX;
			break;
		}
		break;
	case 'M': case 'm':
		if (strncasecmp(buf + 1, "ARK ", 4) == 0)
			return PDB_USER_MARK;
		if (strncasecmp(buf + 1, "ARKNAME ", 6) == 0)
			return PDB_USER_MARKNAME;
		break;
	case 'O': case 'o':
		if (strncasecmp(buf + 1, "BJECT", 5) == 0
		&& (buf[6] == '\0' || buf[6] == '\n' || buf[6] == ' '))
			return PDB_USER_OBJECT;
		break;
	case 'P': case 'p':
		if (strncasecmp(buf + 1, "DBRUN ", 6) == 0)
			return PDB_USER_PDBRUN;
		break;
	case 'R': case 'r':
		if (strncasecmp(buf + 1, "ADIUS ", 6) == 0)
			return PDB_USER_RADIUS;
		break;
	case 'V': case 'v':
		if (strncasecmp(buf + 1, "IEWPORT ", 6) == 0)
			return PDB_USER_VIEWPORT;
		break;
	case 'W': case 'w':
		if (strncasecmp(buf + 1, "INDOW ", 6) == 0)
			return PDB_USER_WINDOW;
		break;
	}
	return PDB_USER;
}

pdb_record
# ifdef __STDC__
pdb_read_string(const char *buffer)
# else
pdb_read_string(buffer)
	char	*buffer;
# endif
{

	pdb_record		r;
	const char		*fmt;
	struct pdb_sheet	*sh;
	pdb_residue		*sha0, *sha1;
	char			record_type[4];
	int			i;
	char			new_buffer[PDB_BUFSIZ];

# if defined(__STDC__) || defined(vms)
	(void) memset(&r, 0, sizeof r);
# else
	bzero((char *) &r, sizeof r);
# endif

# ifdef DEBUG
	printf("%s", buffer);
# endif /* DEBUG */

	/* convert pdb record to C structure */

	for (i = 0; buffer[i] != '\0' && buffer[i] != '\n' && i < 4; i += 1) {
		if (isupper(buffer[i]))
			record_type[i] = _tolower(buffer[i]);
		else
			record_type[i] = buffer[i];
	}
	if (i < 4)
		for (; i < 4; i += 1)
			record_type[i] = ' ';

	r.record_type = PDB_UNKNOWN;
	switch (record_type[0]) {

	case 'a':
		if (STREQN(record_type + 1, "tom", 3))
			r.record_type = PDB_ATOM;
		else if (STREQN(record_type + 1, "uth", 3))
			r.record_type = PDB_AUTHOR;
		else if (STREQN(record_type + 1, "nis", 3))
			r.record_type = PDB_ANISOU;
		else if (STREQN(record_type + 1, "grd", 3))
			r.record_type = PDB_AGRDES;
		else if (STREQN(record_type + 1, "ggr", 3))
			r.record_type = PDB_AGGRGT;
		break;

	case 'c':
		if (STREQN(record_type + 1, "omp", 3))
			r.record_type = PDB_COMPND;
		else if (STREQN(record_type + 1, "rys", 3))
			r.record_type = PDB_CRYST1;
		else if (STREQN(record_type + 1, "one", 3))
			r.record_type = PDB_CONECT;
		else if (STREQN(record_type + 1, "mpd", 3))
			r.record_type = PDB_CMPDES;
		else if (STREQN(record_type + 1, "mpo", 3))
			r.record_type = PDB_CMPONT;
		break;

	case 'e':
		if (STREQN(record_type + 1, "nd ", 3))
			r.record_type = PDB_END;
		else if (STREQN(record_type + 1, "ndm", 3))
			r.record_type = PDB_ENDMDL;
		else if (STREQN(record_type + 1, "xpd", 3))
			r.record_type = PDB_EXPDTA;
		break;

	case 'f':
		if (STREQN(record_type + 1, "tno", 3))
			r.record_type = PDB_FTNOTE;
		else if (STREQN(record_type + 1, "orm", 3))
			r.record_type = PDB_FORMUL;
		break;

	case 'h':
		if (STREQN(record_type + 1, "eta", 3))
			r.record_type = PDB_HETATM;
		else if (STREQN(record_type + 1, "ead", 3))
			r.record_type = PDB_HEADER;
		else if (STREQN(record_type + 1, "et ", 3))
			r.record_type = PDB_HET;
		else if (STREQN(record_type + 1, "eli", 3))
			r.record_type = PDB_HELIX;
		break;

	case 'j':
		if (STREQN(record_type + 1, "rnl", 3))
			r.record_type = PDB_JRNL;
		break;

	case 'm':
		if (STREQN(record_type + 1, "tri", 3))
			r.record_type = PDB_MTRIX;
		else if (STREQN(record_type + 1, "ast", 3))
			r.record_type = PDB_MASTER;
		else if (STREQN(record_type + 1, "ode", 3))
			r.record_type = PDB_MODEL;
		else if (STREQN(record_type + 1, "txd", 3))
			r.record_type = PDB_MTXDES;
		break;

	case 'o':
		if (STREQN(record_type + 1, "bsl", 3))
			r.record_type = PDB_OBSLTE;
		else if (STREQN(record_type + 1, "rig", 3))
			r.record_type = PDB_ORIGX;
		break;

	case 'r':
		if (STREQN(record_type + 1, "ema", 3))
			r.record_type = PDB_REMARK;
		else if (STREQN(record_type + 1, "evd", 3))
			r.record_type = PDB_REVDAT;
		break;

	case 's':
		switch (record_type[1]) {

		case 'c':
			if (STREQN(record_type + 2, "al", 2))
				r.record_type = PDB_SCALE;
			break;

		case 'e':
			if (STREQN(record_type + 2, "qr", 2))
				r.record_type = PDB_SEQRES;
			break;

		case 'h':
			if (STREQN(record_type + 2, "ee", 2))
				r.record_type = PDB_SHEET;
			break;

		case 'i':
			if (STREQN(record_type + 2, "te", 2))
				r.record_type = PDB_SITE;
			else if (STREQN(record_type + 2, "ga", 2))
				r.record_type = PDB_SIGATM;
			else if (STREQN(record_type + 2, "gu", 2))
				r.record_type = PDB_SIGUIJ;
			break;

		case 'o':
			if (STREQN(record_type + 2, "ur", 2))
				r.record_type = PDB_SOURCE;
			break;

		case 'p':
			if (STREQN(record_type + 2, "rs", 2))
				r.record_type = PDB_SPRSDE;
			break;

		case 's':
			if (STREQN(record_type + 2, "bo", 2))
				r.record_type = PDB_SSBOND;
			break;

		case 'y':
			if (STREQN(record_type + 2, "md", 2))
				r.record_type = PDB_SYMDES;
			else if (STREQN(record_type + 2, "mo", 2))
				r.record_type = PDB_SYMOP;
			break;
		}
		break;

	case 't':
		if (STREQN(record_type + 1, "urn", 3))
			r.record_type = PDB_TURN;
		else if (STREQN(record_type + 1, "vec", 3))
			r.record_type = PDB_TVECT;
		else if (STREQN(record_type + 1, "er ", 3))
			r.record_type = PDB_TER;
		else if (STREQN(record_type + 1, "rns", 3))
			r.record_type = PDB_TRNSFM;
		break;

	case 'u':
		if (STREQN(record_type + 1, "ser", 3)) {
			r.record_type = PDB_USER;
			switch (pdb_pdbrun_version) {
			case 1: case 2: case 3: case 4: case 5:
				r.record_type = pdbrun5_type(buffer + 6);
				break;
			case 6:
				r.record_type = pdbrun6_type(buffer + 6);
				break;
			default:
				if (strncasecmp(buffer + 6, "PDBRUN ", 7) == 0)
					r.record_type = PDB_USER_PDBRUN;
				break;
			}
		}
		break;
	}

	if (r.record_type < PDB_USER_PDBRUN)
		fmt = pdb_record_format[r.record_type];
	else if (pdb_pdbrun_version < 6)
		fmt = pdbrun5[r.record_type - PDB_USER_PDBRUN];
	else
		fmt = pdbrun6[r.record_type - PDB_USER_PDBRUN];
	switch (r.record_type) {

	default:
	case PDB_UNKNOWN:
unknown:
		r.record_type = PDB_UNKNOWN;		/* in case of goto */
		(void) pdb_sscanf(buffer, "%72s", r.pdb.unknown.junk);
		break;

	case PDB_AGGRGT:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.aggrgt.serial_num,
				&r.pdb.aggrgt.num_components,
				&r.pdb.aggrgt.cmpont_serial_nums[0],
				&r.pdb.aggrgt.cmpont_serial_nums[1],
				&r.pdb.aggrgt.cmpont_serial_nums[2],
				&r.pdb.aggrgt.cmpont_serial_nums[3],
				&r.pdb.aggrgt.cmpont_serial_nums[4],
				&r.pdb.aggrgt.cmpont_serial_nums[5],
				&r.pdb.aggrgt.cmpont_serial_nums[6],
				&r.pdb.aggrgt.cmpont_serial_nums[7],
				&r.pdb.aggrgt.cmpont_serial_nums[8],
				&r.pdb.aggrgt.cmpont_serial_nums[9],
				&r.pdb.aggrgt.cmpont_serial_nums[10],
				&r.pdb.aggrgt.cmpont_serial_nums[11],
				&r.pdb.aggrgt.cmpont_serial_nums[12],
				&r.pdb.aggrgt.cmpont_serial_nums[13]))
			goto unknown;
		break;
	case PDB_ANISOU:
	case PDB_SIGUIJ:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.anisou.serial_num,
				r.pdb.anisou.name, &r.pdb.anisou.alt_loc,
				r.pdb.anisou.residue.name,
				&r.pdb.anisou.residue.chain_id,
				&r.pdb.anisou.residue.seq_num,
				&r.pdb.anisou.residue.insert_code,
				&r.pdb.anisou.u[0], &r.pdb.anisou.u[1],
				&r.pdb.anisou.u[2], &r.pdb.anisou.u[3],
				&r.pdb.anisou.u[4], &r.pdb.anisou.u[5]))
			goto unknown;
		break;

	case PDB_ATOM:
	case PDB_HETATM:
	case PDB_SIGATM:
		if (0 <= pdb_sscanf(buffer, fmt, &r.pdb.atom.serial_num,
				r.pdb.atom.name, &r.pdb.atom.alt_loc,
				r.pdb.atom.residue.name,
				&r.pdb.atom.residue.chain_id,
				&r.pdb.atom.residue.seq_num,
				&r.pdb.atom.residue.insert_code,
				&r.pdb.atom.x, &r.pdb.atom.y, &r.pdb.atom.z,
				&r.pdb.atom.occupancy, &r.pdb.atom.temp_factor,
				&r.pdb.atom.ftnote_num))
			break;
		/* handle atom serial number overflows */
		if (strncmp(&buffer[6], "*****", 5) != 0)
			goto unknown;
		strncpy(new_buffer, buffer, PDB_BUFSIZ);
		strncpy(&new_buffer[6], "00000", 5);
		if (0 <= pdb_sscanf(new_buffer, fmt, &r.pdb.atom.serial_num,
				r.pdb.atom.name, &r.pdb.atom.alt_loc,
				r.pdb.atom.residue.name,
				&r.pdb.atom.residue.chain_id,
				&r.pdb.atom.residue.seq_num,
				&r.pdb.atom.residue.insert_code,
				&r.pdb.atom.x, &r.pdb.atom.y, &r.pdb.atom.z,
				&r.pdb.atom.occupancy, &r.pdb.atom.temp_factor,
				&r.pdb.atom.ftnote_num)) {
			static int atom_serial_number = 10000;
			r.pdb.atom.serial_num = atom_serial_number++;
			break;
		}
		goto unknown;

	case PDB_AUTHOR:
	case PDB_COMPND:
	case PDB_JRNL:
	case PDB_SOURCE:
	case PDB_EXPDTA:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.author.continuation,
				r.pdb.author.data))
			goto unknown;
		break;

	case PDB_CMPONT:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.cmpont.seq_num,
				r.pdb.cmpont.residues[0].name,
				&r.pdb.cmpont.residues[0].chain_id,
				&r.pdb.cmpont.residues[0].seq_num,
				&r.pdb.cmpont.residues[0].insert_code,
				r.pdb.cmpont.residues[1].name,
				&r.pdb.cmpont.residues[1].chain_id,
				&r.pdb.cmpont.residues[1].seq_num,
				&r.pdb.cmpont.residues[1].insert_code))
			goto unknown;
		break;

	case PDB_CONECT:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.conect.serial_num,
				&r.pdb.conect.covalent[0],
				&r.pdb.conect.covalent[1],
				&r.pdb.conect.covalent[2],
				&r.pdb.conect.covalent[3],
				&r.pdb.conect.bonds[0].hydrogen[0],
				&r.pdb.conect.bonds[0].hydrogen[1],
				&r.pdb.conect.bonds[0].salt,
				&r.pdb.conect.bonds[1].hydrogen[0],
				&r.pdb.conect.bonds[1].hydrogen[1],
				&r.pdb.conect.bonds[1].salt))
			goto unknown;
		break;

	case PDB_CRYST1:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.cryst1.a,
				&r.pdb.cryst1.b, &r.pdb.cryst1.c,
				&r.pdb.cryst1.alpha, &r.pdb.cryst1.beta,
				&r.pdb.cryst1.gamma, r.pdb.cryst1.space_grp,
				&r.pdb.cryst1.z))
			goto unknown;
		break;

	case PDB_END:
	case PDB_ENDMDL:
		break;

	case PDB_FORMUL:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.formul.component,
				r.pdb.formul.het_id, &r.pdb.formul.continuation,
				&r.pdb.formul.exclude, r.pdb.formul.formula))
			goto unknown;
		break;

	case PDB_FTNOTE:
	case PDB_REMARK:
	case PDB_SYMDES:
	case PDB_MTXDES:
	case PDB_CMPDES:
	case PDB_AGRDES:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.ftnote.num,
				r.pdb.ftnote.text))
			goto unknown;
		break;

	case PDB_HEADER:
		if (0 > pdb_sscanf(buffer, fmt, r.pdb.header.class,
				r.pdb.header.date, &r.pdb.header.type,
				r.pdb.header.id))
			goto unknown;
		break;

	case PDB_HELIX:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.helix.serial_num,
				r.pdb.helix.id,
				r.pdb.helix.residues[0].name,
				&r.pdb.helix.residues[0].chain_id,
				&r.pdb.helix.residues[0].seq_num,
				&r.pdb.helix.residues[0].insert_code,
				r.pdb.helix.residues[1].name,
				&r.pdb.helix.residues[1].chain_id,
				&r.pdb.helix.residues[1].seq_num,
				&r.pdb.helix.residues[1].insert_code,
				&r.pdb.helix.class, r.pdb.helix.comment))
			goto unknown;
		break;

	case PDB_HET:
		if (0 > pdb_sscanf(buffer, fmt, r.pdb.het.het_grp.name,
				&r.pdb.het.het_grp.chain_id,
				&r.pdb.het.het_grp.seq_num,
				&r.pdb.het.het_grp.insert_code,
				&r.pdb.het.num_atoms, r.pdb.het.text))
			goto unknown;
		break;

	case PDB_MASTER:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.master.num_remark,
				&r.pdb.master.num_ftnote,
				&r.pdb.master.num_het, &r.pdb.master.num_helix,
				&r.pdb.master.num_sheet, &r.pdb.master.num_turn,
				&r.pdb.master.num_site,
				&r.pdb.master.num_transform,
				&r.pdb.master.num_coordinate,
				&r.pdb.master.num_ter, &r.pdb.master.num_conect,
				&r.pdb.master.num_seqres))
			goto unknown;
		break;

	case PDB_MODEL:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.model.num))
			goto unknown;
		break;

	case PDB_MTRIX:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.mtrix.row_num,
				&r.pdb.mtrix.serial_num, &r.pdb.mtrix.m1,
				&r.pdb.mtrix.m2, &r.pdb.mtrix.m3,
				&r.pdb.mtrix.v, &r.pdb.mtrix.given))
			goto unknown;
		break;

	case PDB_OBSLTE:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.obslte.continuation,
				r.pdb.obslte.date, r.pdb.obslte.old_id,
				r.pdb.obslte.id_map[0], r.pdb.obslte.id_map[1],
				r.pdb.obslte.id_map[2], r.pdb.obslte.id_map[3],
				r.pdb.obslte.id_map[4], r.pdb.obslte.id_map[2],
				r.pdb.obslte.id_map[6], r.pdb.obslte.id_map[7]))
			goto unknown;
		break;

	case PDB_ORIGX:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.origx.row_num,
				&r.pdb.origx.o1, &r.pdb.origx.o2,
				&r.pdb.origx.o3, &r.pdb.origx.t))
			goto unknown;
		break;

	case PDB_REVDAT:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.revdat.modification,
				&r.pdb.revdat.continuation, r.pdb.revdat.date,
				r.pdb.revdat.id, &r.pdb.revdat.mod_type,
				r.pdb.revdat.corrections))
			goto unknown;
		break;

	case PDB_SCALE:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.scale.row_num,
				&r.pdb.scale.s1, &r.pdb.scale.s2,
				&r.pdb.scale.s3, &r.pdb.scale.u))
			goto unknown;
		break;

	case PDB_SEQRES:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.seqres.serial_num,
				&r.pdb.seqres.chain_id, &r.pdb.seqres.count,
				r.pdb.seqres.names[0], r.pdb.seqres.names[1],
				r.pdb.seqres.names[2], r.pdb.seqres.names[3],
				r.pdb.seqres.names[4], r.pdb.seqres.names[5],
				r.pdb.seqres.names[6], r.pdb.seqres.names[7],
				r.pdb.seqres.names[8], r.pdb.seqres.names[9],
				r.pdb.seqres.names[10], r.pdb.seqres.names[11],
				r.pdb.seqres.names[12]))
			goto unknown;
		break;

	case PDB_SHEET:
		sh = &r.pdb.sheet;
		sha0 = &sh->atoms[0].residue;
		sha1 = &sh->atoms[1].residue;
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.sheet.strand_num,
				sh->id, &r.pdb.sheet.count,
				sh->residues[0].name, &sh->residues[0].chain_id,
				&sh->residues[0].seq_num,
				&sh->residues[0].insert_code,
				sh->residues[1].name, &sh->residues[1].chain_id,
				&sh->residues[1].seq_num,
				&sh->residues[1].insert_code, &sh->sense,
				sh->atoms[0].name, sha0->name, &sha0->chain_id,
				&sha0->seq_num, &sha0->insert_code,
				sh->atoms[1].name, sha1->name,
				&sha1->chain_id, &sha1->seq_num,
				&sha1->insert_code))
			goto unknown;
		break;

	case PDB_SITE:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.site.seq_num,
				r.pdb.site.id, &r.pdb.site.count,
				r.pdb.site.residues[0].name,
				&r.pdb.site.residues[0].chain_id,
				&r.pdb.site.residues[0].seq_num,
				&r.pdb.site.residues[0].insert_code,
				r.pdb.site.residues[1].name,
				&r.pdb.site.residues[1].chain_id,
				&r.pdb.site.residues[1].seq_num,
				&r.pdb.site.residues[1].insert_code,
				r.pdb.site.residues[2].name,
				&r.pdb.site.residues[2].chain_id,
				&r.pdb.site.residues[2].seq_num,
				&r.pdb.site.residues[2].insert_code,
				r.pdb.site.residues[3].name,
				&r.pdb.site.residues[3].chain_id,
				&r.pdb.site.residues[3].seq_num,
				&r.pdb.site.residues[3].insert_code))
			goto unknown;
		break;

	case PDB_SPRSDE:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.sprsde.continuation,
				r.pdb.sprsde.date, r.pdb.sprsde.id,
				r.pdb.sprsde.supersede[0],
				r.pdb.sprsde.supersede[1],
				r.pdb.sprsde.supersede[2],
				r.pdb.sprsde.supersede[3],
				r.pdb.sprsde.supersede[4],
				r.pdb.sprsde.supersede[5],
				r.pdb.sprsde.supersede[6],
				r.pdb.sprsde.supersede[7]))
			goto unknown;
		break;

	case PDB_SSBOND:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.ssbond.seq_num,
				r.pdb.ssbond.residues[0].name,
				&r.pdb.ssbond.residues[0].chain_id,
				&r.pdb.ssbond.residues[0].seq_num,
				&r.pdb.ssbond.residues[0].insert_code,
				r.pdb.ssbond.residues[1].name,
				&r.pdb.ssbond.residues[1].chain_id,
				&r.pdb.ssbond.residues[1].seq_num,
				&r.pdb.ssbond.residues[1].insert_code,
				r.pdb.ssbond.comment))
			goto unknown;
		break;

	case PDB_SYMOP:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.symop.row_num,
				&r.pdb.symop.serial_num, &r.pdb.symop.s1,
				&r.pdb.symop.s2, &r.pdb.symop.s3,
				&r.pdb.symop.t))
			goto unknown;
		break;

	case PDB_TER:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.ter.serial_num,
				r.pdb.ter.residue.name,
				&r.pdb.ter.residue.chain_id,
				&r.pdb.ter.residue.seq_num,
				&r.pdb.ter.residue.insert_code))
			goto unknown;
		break;

	case PDB_TRNSFM:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.trnsfm.result_serial_num,
				&r.pdb.trnsfm.apply_serial_num,
				&r.pdb.trnsfm.source_serial_num))
			goto unknown;
		break;

	case PDB_TURN:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.turn.seq_num,
				r.pdb.turn.id, r.pdb.turn.residues[0].name,
				&r.pdb.turn.residues[0].chain_id,
				&r.pdb.turn.residues[0].seq_num,
				&r.pdb.turn.residues[0].insert_code,
				r.pdb.turn.residues[1].name,
				&r.pdb.turn.residues[1].chain_id,
				&r.pdb.turn.residues[1].seq_num,
				&r.pdb.turn.residues[1].insert_code,
				r.pdb.turn.comment))
			goto unknown;
		break;

	case PDB_TVECT:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.tvect.serial_num,
				&r.pdb.tvect.t1, &r.pdb.tvect.t2,
				&r.pdb.tvect.t3, r.pdb.tvect.comment))
			goto unknown;
		break;

user:
		r.record_type = PDB_USER;
		fmt = pdb_record_format[r.record_type];
	case PDB_USER:
		if (0 > pdb_sscanf(buffer, fmt, r.pdb.user.subtype,
				r.pdb.user.text))
			goto unknown;
		break;

	case PDB_USER_PDBRUN:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_pdbrun.version))
			goto user;
		pdb_pdbrun_version = r.pdb.user_pdbrun.version;
		break;

	case PDB_USER_EYEPOS:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_eyepos.xyz[0],
				&r.pdb.user_eyepos.xyz[1],
				&r.pdb.user_eyepos.xyz[2]))
			goto user;
		break;

	case PDB_USER_ATPOS:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_atpos.xyz[0],
				&r.pdb.user_atpos.xyz[1],
				&r.pdb.user_atpos.xyz[2]))
			goto user;
		break;

	case PDB_USER_WINDOW:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_window.left,
				&r.pdb.user_window.right,
				&r.pdb.user_window.bottom,
				&r.pdb.user_window.top,
				&r.pdb.user_window.hither,
				&r.pdb.user_window.yon))
			goto user;
		break;

	case PDB_USER_FOCUS:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_focus.focus))
			goto user;
		break;

	case PDB_USER_VIEWPORT:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_viewport.xmin,
				&r.pdb.user_viewport.xmax,
				&r.pdb.user_viewport.ymin,
				&r.pdb.user_viewport.ymax))
			goto user;
		break;

	case PDB_USER_BGCOLOR:
		if (pdb_pdbrun_version < 6) {
			if (0 > sscanf(buffer, fmt, &r.pdb.user_bgcolor.rgb[0],
					&r.pdb.user_bgcolor.rgb[1],
					&r.pdb.user_bgcolor.rgb[2]))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt,
				&r.pdb.user_bgcolor.rgb[0],
				&r.pdb.user_bgcolor.rgb[1],
				&r.pdb.user_bgcolor.rgb[2]))
			goto user;
		break;

	case PDB_USER_ANGLE:
		if (pdb_pdbrun_version < 6) {
			if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_angle.which,
					&r.pdb.user_angle.atom0,
					&r.pdb.user_angle.atom1,
					&r.pdb.user_angle.atom2,
					&r.pdb.user_angle.atom3,
					&r.pdb.user_angle.angle))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_angle.atom0,
				&r.pdb.user_angle.atom1,
				&r.pdb.user_angle.atom2,
				&r.pdb.user_angle.atom3,
				&r.pdb.user_angle.angle))
			goto user;
		break;

	case PDB_USER_DISTANCE:
		if (pdb_pdbrun_version < 6) {
			if (0 > pdb_sscanf(buffer, fmt,
					&r.pdb.user_distance.which,
					&r.pdb.user_distance.atom0,
					&r.pdb.user_distance.atom1,
					&r.pdb.user_distance.distance))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt,
				&r.pdb.user_distance.atom0,
				&r.pdb.user_distance.atom1,
				&r.pdb.user_distance.distance))
			goto user;
		break;

	case PDB_USER_FILE:
		if (pdb_pdbrun_version < 6) {
			if (0 > pdb_sscanf(buffer, fmt,
						r.pdb.user_file.filename))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_file.model,
						r.pdb.user_file.filename))
			goto user;
		break;

	case PDB_USER_MARKNAME:
		if (0 > pdb_sscanf(buffer, fmt, r.pdb.user_markname.markname))
			goto user;
		break;

	case PDB_USER_MARK:
		if (0 > pdb_sscanf(buffer, fmt, r.pdb.user_mark.markname))
			goto user;
		break;

	case PDB_USER_CNAME:
		if (pdb_pdbrun_version < 6) {
			if (0 > sscanf(buffer, fmt, r.pdb.user_cname.name,
					&r.pdb.user_cname.rgb[0],
					&r.pdb.user_cname.rgb[1],
					&r.pdb.user_cname.rgb[2]))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_cname.rgb[0],
				&r.pdb.user_cname.rgb[1],
				&r.pdb.user_cname.rgb[2],
				r.pdb.user_cname.name))
			goto user;
		break;

	case PDB_USER_COLOR:
		if (pdb_pdbrun_version < 6) {
			if (0 > sscanf(buffer, fmt, r.pdb.user_color.spec,
					&r.pdb.user_color.rgb[0],
					&r.pdb.user_color.rgb[1],
					&r.pdb.user_color.rgb[2]))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_color.rgb[0],
				&r.pdb.user_color.rgb[1],
				&r.pdb.user_color.rgb[2],
				r.pdb.user_color.spec))
			goto user;
		break;

	case PDB_USER_RADIUS:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_radius.radius))
			goto user;
		break;

	case PDB_USER_OBJECT:
		if (pdb_pdbrun_version < 6) {
			if (0 > pdb_sscanf(buffer, fmt,
					&r.pdb.user_object.model))
				goto user;
		}
		break;

	case PDB_USER_ENDOBJ:
		if (pdb_pdbrun_version < 6) {
			if (0 > pdb_sscanf(buffer, fmt,
					&r.pdb.user_endobj.model))
				goto user;
		}
		break;

	case PDB_USER_CHAIN:
		if (pdb_pdbrun_version < 6) {
			if (0 > sscanf(buffer, fmt, &r.pdb.user_chain.atom0,
					&r.pdb.user_chain.atom1))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_chain.atom0,
				&r.pdb.user_chain.atom1))
			goto user;
		break;

	case PDB_USER_GFX_BEGIN:
		if (0 > pdb_sscanf(buffer, fmt, r.pdb.user_gfx_begin.unknown))
			goto user;
		r.pdb.user_gfx_begin.primitive
			= pdb_gfx_type(r.pdb.user_gfx_begin.unknown);
		break;

	case PDB_USER_GFX_END:
		break;

	case PDB_USER_GFX_COLOR:
		if (pdb_pdbrun_version < 6) {
			if (0 > sscanf(buffer, fmt, r.pdb.user_gfx_color.spec,
					&r.pdb.user_gfx_color.rgb[0],
					&r.pdb.user_gfx_color.rgb[1],
					&r.pdb.user_gfx_color.rgb[2]))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt,
				&r.pdb.user_gfx_color.rgb[0],
				&r.pdb.user_gfx_color.rgb[1],
				&r.pdb.user_gfx_color.rgb[2],
				r.pdb.user_gfx_color.spec))
			goto user;
		break;

	case PDB_USER_GFX_NORMAL:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_gfx_normal.xyz[0],
				&r.pdb.user_gfx_normal.xyz[1],
				&r.pdb.user_gfx_normal.xyz[2]))
			goto user;
		break;

	case PDB_USER_GFX_VERTEX:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_gfx_vertex.xyz[0],
				&r.pdb.user_gfx_vertex.xyz[1],
				&r.pdb.user_gfx_vertex.xyz[2]))
			goto user;
		break;

	case PDB_USER_GFX_FONT:
		if (pdb_pdbrun_version < 6) {
			if (0 > sscanf(buffer, fmt, r.pdb.user_gfx_font.name,
					&r.pdb.user_gfx_font.size))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt,
				&r.pdb.user_gfx_font.size,
				r.pdb.user_gfx_font.name))
			goto user;
		break;

	case PDB_USER_GFX_TEXTPOS:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_gfx_textpos.xyz[0],
				&r.pdb.user_gfx_textpos.xyz[1],
				&r.pdb.user_gfx_textpos.xyz[2]))
			goto user;
		break;

	case PDB_USER_GFX_LABEL:
		if (pdb_pdbrun_version < 6) {
			if (0 > sscanf(buffer, fmt,
					&r.pdb.user_gfx_label.xyz[0],
					&r.pdb.user_gfx_label.xyz[1],
					&r.pdb.user_gfx_label.xyz[2],
					r.pdb.user_gfx_label.text))
				goto user;
		} else if (0 > pdb_sscanf(buffer, fmt,
				r.pdb.user_gfx_label.text))
			goto user;
		/* TODO: process text? */
		break;

	case PDB_USER_GFX_MOVE:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_gfx_move.xyz[0],
				&r.pdb.user_gfx_move.xyz[1],
				&r.pdb.user_gfx_move.xyz[2]))
			goto user;
		break;

	case PDB_USER_GFX_DRAW:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_gfx_draw.xyz[0],
				&r.pdb.user_gfx_draw.xyz[1],
				&r.pdb.user_gfx_draw.xyz[2]))
			goto user;
		break;

	case PDB_USER_GFX_MARKER:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_gfx_marker.xyz[0],
				&r.pdb.user_gfx_marker.xyz[1],
				&r.pdb.user_gfx_marker.xyz[2]))
			goto user;
		break;

	case PDB_USER_GFX_POINT:
		if (0 > pdb_sscanf(buffer, fmt, &r.pdb.user_gfx_point.xyz[0],
				&r.pdb.user_gfx_point.xyz[1],
				&r.pdb.user_gfx_point.xyz[2]))
			goto user;
		break;
	}
	
	return r;
}

# ifdef vms
pdb_read_dummy()
{
	pdb_fmt_dummy();
}
# endif
