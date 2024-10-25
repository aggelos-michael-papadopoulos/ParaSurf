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
 *	$Id: pdb.h,v 2.11 1995/01/20 01:20:36 gregc Exp $
 *
 *	Based on Brookhaven National Laboratory Protein Data Bank, March 1989
 *
 *	C structure declarations
 */

#ifndef PDB_H
# define	PDB_H

# include	<stdio.h>

# ifdef __cplusplus
extern "C" {
# endif

# define	PDB_RECLEN	80		/* PDB record length */
# define	PDB_BUFSIZ	PDB_RECLEN + 2	/* + '\n' + '\0' */		

# define	PDB_PDBRUN_VERSION	6

# define	PDB_UNKNOWN	0

/* records originally in alphabetical order */

# define	PDB_ANISOU	1
# define	PDB_ATOM	2
# define	PDB_AUTHOR	3
# define	PDB_COMPND	4
# define	PDB_CONECT	5
# define	PDB_CRYST1	6
# define	PDB_END		7
# define	PDB_FORMUL	8
# define	PDB_FTNOTE	9
# define	PDB_HEADER	10
# define	PDB_HELIX	11
# define	PDB_HET		12
# define	PDB_HETATM	13
# define	PDB_JRNL	14
# define	PDB_MASTER	15
# define	PDB_MTRIX	16
# define	PDB_OBSLTE	17
# define	PDB_ORIGX	18
# define	PDB_REMARK	19
# define	PDB_REVDAT	20
# define	PDB_SCALE	21
# define	PDB_SEQRES	22
# define	PDB_SHEET	23
# define	PDB_SIGATM	24
# define	PDB_SIGUIJ	25
# define	PDB_SITE	26
# define	PDB_SOURCE	27
# define	PDB_SPRSDE	28
# define	PDB_SSBOND	29
# define	PDB_TER		30
# define	PDB_TURN	31
# define	PDB_TVECT	32
# define	PDB_USER	33
# define	PDB_MODEL	34
# define	PDB_ENDMDL	35
# define	PDB_EXPDTA	36
# define	PDB_SYMDES	37
# define	PDB_SYMOP	38
# define	PDB_MTXDES	39
# define	PDB_CMPDES	40
# define	PDB_CMPONT	41
# define	PDB_TRNSFM	42
# define	PDB_AGRDES	43
# define	PDB_AGGRGT	44

# define	PDB_NUM_R	45

# define	PDB_USER_PDBRUN		0x100
# define	PDB_USER_EYEPOS		0x101
# define	PDB_USER_ATPOS		0x102
# define	PDB_USER_WINDOW		0x103
# define	PDB_USER_FOCUS		0x104
# define	PDB_USER_VIEWPORT	0x105
# define	PDB_USER_BGCOLOR	0x106
# define	PDB_USER_ANGLE		0x107
# define	PDB_USER_DISTANCE	0x108
# define	PDB_USER_FILE		0x109
# define	PDB_USER_MARKNAME	0x10a
# define	PDB_USER_MARK		0x10b
# define	PDB_USER_CNAME		0x10c
# define	PDB_USER_COLOR		0x10d
# define	PDB_USER_RADIUS		0x10e
# define	PDB_USER_OBJECT		0x10f
# define	PDB_USER_ENDOBJ		0x110
# define	PDB_USER_CHAIN		0x111
# define	PDB_USER_GFX_BEGIN	0x112
# define	PDB_USER_GFX_END	0x113
# define	PDB_USER_GFX_COLOR	0x114
# define	PDB_USER_GFX_NORMAL	0x115
# define	PDB_USER_GFX_VERTEX	0x116
# define	PDB_USER_GFX_FONT	0x117
# define	PDB_USER_GFX_TEXTPOS	0x118
# define	PDB_USER_GFX_LABEL	0x119
# define	PDB_USER_GFX_MOVE	0x11a	/* obsolete */
# define	PDB_USER_GFX_DRAW	0x11b	/* obsolete */
# define	PDB_USER_GFX_MARKER	0x11c	/* obsolete */
# define	PDB_USER_GFX_POINT	0x11d	/* obsolete */

# define	PDB_NUM_USER_R		(PDB_USER_GFX_POINT-PDB_USER_PDBRUN+1)

# define	PDB_GFX_UNKNOWN		0x0
# define	PDB_GFX_POINTS		0x1
# define	PDB_GFX_MARKERS		0x2
# define	PDB_GFX_LINES		0x3
# define	PDB_GFX_LINE_STRIP	0x4
# define	PDB_GFX_LINE_LOOP	0x5
# define	PDB_GFX_TRIANGLES	0x6
# define	PDB_GFX_TRIANGLE_STRIP	0x7
# define	PDB_GFX_TRIANGLE_FAN	0x8
# define	PDB_GFX_QUADS		0x9
# define	PDB_GFX_QUAD_STRIP	0xa
# define	PDB_GFX_POLYGON		0xb

typedef char	pdb_date[10];
typedef char	pdb_aname[5];		/* atom name - NO2* */
typedef char	pdb_rname[5];		/* residue name - ALA */
typedef char	pdb_pname[5];		/* pdb name - 9lyz */
typedef char	pdb_id[4];		/* generic short id field */
typedef double	pdb_float;		/* size of floating point */

typedef struct {			/* residue info */
	pdb_rname	name;
	char		chain_id;
	int		seq_num;
	char		insert_code;
} pdb_residue;

/*
 *	structures declarations for each record type
 */

struct pdb_unknown {
	char	junk[81];
};
struct pdb_aggrgt {
	int	serial_num;
	int	num_components;
	int	cmpont_serial_nums[14];
};
# define	pdb_agrdes	pdb_ftnote
struct pdb_anisou {
	int		serial_num;
	pdb_aname	name;
	char		alt_loc;
	pdb_residue	residue;
	int		u[6];
};
struct pdb_atom {
	int		serial_num;
	pdb_aname	name;
	char		alt_loc;
	pdb_residue	residue;
	pdb_float	x, y, z;
	pdb_float	occupancy, temp_factor;
	int		ftnote_num;
};
struct pdb_author {
	char	data[61];
	char	continuation;
};
# define	pdb_cmpdes	pdb_ftnote
struct pdb_cmpont {
	int		seq_num;
	pdb_residue	residues[2];
};
# define	pdb_compnd	pdb_author
struct pdb_conect {
	int	serial_num;
	int	covalent[4];
	struct {
		int	hydrogen[2];
		int	salt;
	} bonds[2];
};
struct pdb_cryst1 {
	pdb_float	a, b, c;
	pdb_float	alpha, beta, gamma;
	char		space_grp[12];
	int		z;
};
/* no structure for PDB_END */
/* no structure for PDB_ENDMDL */
# define	pdb_expdta	pdb_author
struct pdb_formul {
	int		component;
	pdb_rname	het_id;
	int		continuation;
	char		exclude;	/* * to exclude */
	char		formula[52];
};
struct pdb_ftnote {
	int	num;
	char	text[60];
};
struct pdb_header {
	char		class[41];
	pdb_date	date;
	pdb_pname	id;
	char		type;
};
struct pdb_helix {
	int		serial_num;
	pdb_id		id;
	pdb_residue	residues[2];
	int		class;
	char		comment[31];
};
struct pdb_het {
	pdb_residue	het_grp;
	int		num_atoms;
	char		text[41];
};
# define	pdb_hetatm	pdb_atom
# define	pdb_jrnl	pdb_author
struct pdb_master {
	int	num_remark;
	int	num_ftnote;
	int	num_het;
	int	num_helix;
	int	num_sheet;
	int	num_turn;
	int	num_site;
	int	num_transform;
	int	num_coordinate;
	int	num_ter;
	int	num_conect;
	int	num_seqres;
};
struct pdb_model {
	int	num;
};
struct pdb_mtrix {
	int		row_num;
	int		serial_num;
	pdb_float	m1, m2, m3, v;
	int		given;
};
# define	pdb_mtxdes	pdb_ftnote
struct pdb_obslte {
	int		continuation;
	pdb_date	date;
	pdb_pname	old_id;
	pdb_pname	id_map[8];
};
struct pdb_origx {
	int		row_num;
	pdb_float	o1, o2, o3, t;
};
# define	pdb_remark	pdb_ftnote
struct pdb_revdat {
	int		modification;
	int		continuation;
	pdb_date	date;
	char		id[8];
	char		mod_type;
	char		corrections[31];
};
struct pdb_scale {
	int		row_num;
	pdb_float	s1, s2, s3, u;
};
struct pdb_seqres {
	int		serial_num;
	char		chain_id;
	int		count;
	pdb_rname	names[13];
};
struct pdb_sheet {
	int		strand_num;
	pdb_id		id;
	int		count;
	pdb_residue	residues[2];
	int		sense;
	struct {
		pdb_aname	name;
		pdb_residue	residue;
	} atoms[2];
};
# define	pdb_sigatm	pdb_atom
# define	pdb_siguij	pdb_anisou
struct pdb_site {
	int		seq_num;
	pdb_id		id;
	int		count;
	pdb_residue	residues[4];
};
# define	pdb_source	pdb_author
struct pdb_sprsde {
	int		continuation;
	pdb_date	date;
	pdb_pname	id;
	pdb_pname	supersede[8];
};
struct pdb_ssbond {
	int		seq_num;
	pdb_residue	residues[2];
	char		comment[31];
};
# define	pdb_symdes	pdb_ftnote
struct pdb_symop {
	int		row_num;
	int		serial_num;
	pdb_float	s1, s2, s3, t;
};
struct pdb_ter {
	int		serial_num;
	pdb_residue	residue;
};
struct pdb_trnsfm {
	int		result_serial_num;
	int		apply_serial_num;
	int		source_serial_num;
};
struct pdb_turn {
	int		seq_num;
	pdb_id		id;
	pdb_residue	residues[2];
	char		comment[31];
};
struct pdb_tvect {
	int		serial_num;
	pdb_float	t1, t2, t3;
	char		comment[31];
};
struct pdb_user {
	char	subtype[3];
	char	text[67];
};
struct pdb_user_pdbrun {
	int	version;
};
struct pdb_user_eyepos {
	pdb_float	xyz[3];
};
# define	pdb_user_atpos	pdb_user_eyepos
struct pdb_user_window {
	pdb_float	left, right, bottom, top, hither, yon;
};
struct pdb_user_focus {
	pdb_float	focus;
};
struct pdb_user_viewport {
	pdb_float	xmin, xmax, ymin, ymax;
};
struct pdb_user_bgcolor {
	pdb_float	rgb[3];
};
struct pdb_user_angle {
	int		atom0, atom1, atom2, atom3;
	pdb_float	angle;
	int		which;			/* version 5 -- obsolete */
};
struct pdb_user_distance {
	int		atom0, atom1;
	pdb_float	distance;
	int		which;			/* version 5 -- obsolete */
};
struct pdb_user_file {
	char		filename[62];		/* 57 in version 6 */
	int		model;			/* not in version 5 */
};
struct pdb_user_markname {
	char		markname[58];
};
# define	pdb_user_mark	pdb_user_markname
struct pdb_user_cname {
	pdb_float	rgb[3];
	char		name[39];
};
struct pdb_user_color {
	pdb_float	rgb[3];
	char		spec[39];
};
struct pdb_user_radius {
	pdb_float	radius;
};
struct pdb_user_object {
	int		model;			/* version 5 -- obsolete */
};
struct pdb_user_endobj {
	int		model;			/* version 5 -- obsolete */
};
struct pdb_user_chain {
	int		atom0, atom1;
};
struct pdb_user_gfx_begin {			/* not in version 5 */
	int		primitive;
	char		unknown[33];
};
/* no structure for USER  GFX END */
# define	pdb_user_gfx_color	pdb_user_color
struct pdb_user_gfx_normal {
	pdb_float	xyz[3];
};
# define	pdb_user_gfx_vertex	pdb_user_gfx_normal
struct pdb_user_gfx_font {
	int	size;
	char	name[54];
};
struct pdb_user_gfx_textpos {			/* not in version 5 */
	pdb_float	xyz[3];
};
struct pdb_user_gfx_label {
	pdb_float	xyz[3];			/* version 5 -- obsolete */
	char		text[57];		/* 27 in version 5 */
};
struct pdb_user_gfx_move {			/* version 5 -- obsolete */
	pdb_float	xyz[3];
};
# define	pdb_user_gfx_draw	pdb_user_gfx_move	/* version 5 -- obsolete */
# define	pdb_user_gfx_marker	pdb_user_gfx_move	/* "" */
# define	pdb_user_gfx_point	pdb_user_gfx_move	/* "" */

typedef struct pdb_record {
	int	record_type;
	union	{
		struct pdb_unknown	unknown;
		struct pdb_agrdes	agrdes;
		struct pdb_aggrgt	aggrgt;
		struct pdb_anisou	anisou;
		struct pdb_atom		atom;
		struct pdb_author	author;
		struct pdb_cmpdes	cmpdes;
		struct pdb_cmpont	cmpont;
		struct pdb_compnd	compnd;
		struct pdb_conect	conect;
		struct pdb_cryst1	cryst1;
		/* no pdb_end structure */
		/* no pdb_endmdl structure */
		struct pdb_expdta	expdta;
		struct pdb_formul	formul;
		struct pdb_ftnote	ftnote;
		struct pdb_header	header;
		struct pdb_helix	helix;
		struct pdb_het		het;
		struct pdb_hetatm	hetatm;
		struct pdb_jrnl		jrnl;
		struct pdb_master	master;
		struct pdb_model	model;
		struct pdb_mtrix	mtrix;
		struct pdb_mtxdes	mtxdes;
		struct pdb_obslte	obslte;
		struct pdb_origx	origx;
		struct pdb_remark	remark;
		struct pdb_revdat	revdat;
		struct pdb_scale	scale;
		struct pdb_seqres	seqres;
		struct pdb_sheet	sheet;
		struct pdb_sigatm	sigatm;
		struct pdb_siguij	siguij;
		struct pdb_site		site;
		struct pdb_source	source;
		struct pdb_sprsde	sprsde;
		struct pdb_ssbond	ssbond;
		struct pdb_symdes	symdes;
		struct pdb_symop	symop;
		struct pdb_ter		ter;
		struct pdb_trnsfm	trnsfm;
		struct pdb_turn		turn;
		struct pdb_tvect	tvect;
		struct pdb_user		user;
		struct pdb_user_pdbrun	user_pdbrun;
		struct pdb_user_eyepos	user_eyepos;
		struct pdb_user_atpos	user_atpos;
		struct pdb_user_window	user_window;
		struct pdb_user_focus	user_focus;
		struct pdb_user_viewport	user_viewport;
		struct pdb_user_bgcolor	user_bgcolor;
		struct pdb_user_angle	user_angle;
		struct pdb_user_distance	user_distance;
		struct pdb_user_file	user_file;
		struct pdb_user_markname	user_markname;
		struct pdb_user_mark	user_mark;
		struct pdb_user_cname	user_cname;
		struct pdb_user_color	user_color;
		struct pdb_user_radius	user_radius;
		struct pdb_user_object	user_object;
		struct pdb_user_endobj	user_endobj;
		struct pdb_user_chain	user_chain;
		struct pdb_user_gfx_begin	user_gfx_begin;
		struct pdb_user_gfx_color	user_gfx_color;
		struct pdb_user_gfx_normal	user_gfx_normal;
		struct pdb_user_gfx_vertex	user_gfx_vertex;
		struct pdb_user_gfx_font	user_gfx_font;
		struct pdb_user_gfx_textpos	user_gfx_textpos;
		struct pdb_user_gfx_label	user_gfx_label;
		struct pdb_user_gfx_move	user_gfx_move;
		struct pdb_user_gfx_draw	user_gfx_draw;
		struct pdb_user_gfx_marker	user_gfx_marker;
		struct pdb_user_gfx_point	user_gfx_point;
	} pdb;
} pdb_record;

# if defined(__STDC__) || defined(__cplusplus)
extern pdb_record	pdb_read_record(FILE *);
extern pdb_record	pdb_read_string(const char *);
extern void		pdb_write_record(FILE *, const pdb_record *, const char *, int);
extern void		pdb_write_string(char *, const pdb_record *);
# else
extern pdb_record	pdb_read_record();
extern pdb_record	pdb_read_string();
extern void		pdb_write_record();
extern void		pdb_write_string();
# endif

# ifdef __cplusplus
}
# endif
#endif /* PDB_H */
