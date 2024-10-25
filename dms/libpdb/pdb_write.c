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
 *	$Id: pdb_write.c,v 2.11 1995/01/20 01:20:36 gregc Exp $
 *
 *	subroutine for writing PDB format files
 *
 */

/* LINTLIBRARY */

# include	<stdio.h>
# include	<ctype.h>
# include	"pdb_int.h"

static const char * const pdb_record_format[PDB_NUM_R] = {
#include "write_format.i"
};

static char const * const pdbrun5[] = {
#include "pdbrun5_write.i"
};

static char const * const pdbrun6[] = {
#include "pdbrun6_write.i"
};

/*
 *	for each pdb record type there is a format reading in the
 *	record values and for printing them out.
 *
 *	The actual format of a line written, is the print format
 *	followed by blank padding to 72 characters, followed by
 *	8 characters of file and line information.
 */

void
# ifdef __STDC__
pdb_write_record(FILE *f, const pdb_record *r, const char *name, int line_num)
# else
pdb_write_record(f, r, name, line_num)
	FILE		*f;
	pdb_record	*r;
	char		*name;		/* if NULL, don't print */
	int		line_num;	/* if name == NULL, don't print */
# endif
{
	char	buffer[PDB_BUFSIZ];

	pdb_write_string(buffer, r);
	if (name == NULL)
		(void) fprintf(f, "%s\n", buffer);
	else if (line_num >= 10000)
		(void) fprintf(f, "%-72.72s%-4.4s%04d\n", buffer, name,
							line_num % 10000);
	else
		(void) fprintf(f, "%-72.72s%-4.4s%4d\n", buffer, name,
								line_num);
}

void
# ifdef __STDC__
pdb_write_string(char *buffer, const pdb_record *r)
# else
pdb_write_string(buffer, r)
	char		*buffer;
	pdb_record	*r;
# endif
{
	const char		*fmt;
	const struct pdb_sheet	*sh;
	const pdb_residue	*shr0, *shr1, *sha0, *sha1;
	char			*s, *t;

	/* convert C structure to pdb record */


	if (r->record_type < PDB_USER_PDBRUN)
		fmt = pdb_record_format[r->record_type];
	else if (pdb_pdbrun_version < 6)
		fmt = pdbrun5[r->record_type - PDB_USER_PDBRUN];
	else
		fmt = pdbrun6[r->record_type - PDB_USER_PDBRUN];
	switch (r->record_type) {

	case PDB_UNKNOWN:
		pdb_sprintf(buffer, fmt, r->pdb.unknown.junk);
		break;

	case PDB_AGGRGT:
		pdb_sprintf(buffer, fmt, r->pdb.aggrgt.serial_num,
			r->pdb.aggrgt.num_components,
			r->pdb.aggrgt.cmpont_serial_nums[0],
			r->pdb.aggrgt.cmpont_serial_nums[1],
			r->pdb.aggrgt.cmpont_serial_nums[2],
			r->pdb.aggrgt.cmpont_serial_nums[3],
			r->pdb.aggrgt.cmpont_serial_nums[4],
			r->pdb.aggrgt.cmpont_serial_nums[5],
			r->pdb.aggrgt.cmpont_serial_nums[6],
			r->pdb.aggrgt.cmpont_serial_nums[7],
			r->pdb.aggrgt.cmpont_serial_nums[8],
			r->pdb.aggrgt.cmpont_serial_nums[9],
			r->pdb.aggrgt.cmpont_serial_nums[10],
			r->pdb.aggrgt.cmpont_serial_nums[11],
			r->pdb.aggrgt.cmpont_serial_nums[12],
			r->pdb.aggrgt.cmpont_serial_nums[13]);
		break;

	case PDB_ANISOU:
	case PDB_SIGUIJ:
		pdb_sprintf(buffer, fmt, r->pdb.anisou.serial_num,
			r->pdb.anisou.name, r->pdb.anisou.alt_loc,
			r->pdb.anisou.residue.name,
			r->pdb.anisou.residue.chain_id,
			r->pdb.anisou.residue.seq_num,
			r->pdb.anisou.residue.insert_code,
			r->pdb.anisou.u[0], r->pdb.anisou.u[1],
			r->pdb.anisou.u[2], r->pdb.anisou.u[3],
			r->pdb.anisou.u[4], r->pdb.anisou.u[5]);
		break;

	case PDB_ATOM:
	case PDB_HETATM:
	case PDB_SIGATM:
		pdb_sprintf(buffer, fmt, r->pdb.atom.serial_num,
			r->pdb.atom.name, r->pdb.atom.alt_loc,
			r->pdb.atom.residue.name,
			r->pdb.atom.residue.chain_id,
			r->pdb.atom.residue.seq_num,
			r->pdb.atom.residue.insert_code,
			r->pdb.atom.x, r->pdb.atom.y, r->pdb.atom.z,
			r->pdb.atom.occupancy, r->pdb.atom.temp_factor,
			r->pdb.atom.ftnote_num);
		break;

	case PDB_AUTHOR:
	case PDB_COMPND:
	case PDB_JRNL:
	case PDB_SOURCE:
	case PDB_EXPDTA:
		pdb_sprintf(buffer, fmt, r->pdb.author.continuation,
			r->pdb.author.data);
		break;

	case PDB_CONECT:
		pdb_sprintf(buffer, fmt, r->pdb.conect.serial_num,
			r->pdb.conect.covalent[0], r->pdb.conect.covalent[1],
			r->pdb.conect.covalent[2], r->pdb.conect.covalent[3],
			r->pdb.conect.bonds[0].hydrogen[0],
			r->pdb.conect.bonds[0].hydrogen[1],
			r->pdb.conect.bonds[0].salt,
			r->pdb.conect.bonds[1].hydrogen[0],
			r->pdb.conect.bonds[1].hydrogen[1],
			r->pdb.conect.bonds[1].salt);
		break;

	case PDB_CMPONT:
		pdb_sprintf(buffer, fmt, r->pdb.cmpont.seq_num,
			r->pdb.cmpont.residues[0].name,
			r->pdb.cmpont.residues[0].chain_id,
			r->pdb.cmpont.residues[0].seq_num,
			r->pdb.cmpont.residues[0].insert_code,
			r->pdb.cmpont.residues[1].name,
			r->pdb.cmpont.residues[1].chain_id,
			r->pdb.cmpont.residues[1].seq_num,
			r->pdb.cmpont.residues[1].insert_code);
		break;

	case PDB_CRYST1:
		pdb_sprintf(buffer, fmt, r->pdb.cryst1.a, r->pdb.cryst1.b,
			r->pdb.cryst1.c, r->pdb.cryst1.alpha,
			r->pdb.cryst1.beta, r->pdb.cryst1.gamma,
			r->pdb.cryst1.space_grp, r->pdb.cryst1.z);
		break;

	case PDB_END:
	case PDB_ENDMDL:
		pdb_sprintf(buffer, fmt);
		break;

	case PDB_FORMUL:
		pdb_sprintf(buffer, fmt, r->pdb.formul.component,
			r->pdb.formul.het_id, r->pdb.formul.continuation,
			r->pdb.formul.exclude, r->pdb.formul.formula);
		break;

	case PDB_FTNOTE:
	case PDB_REMARK:
	case PDB_SYMDES:
	case PDB_MTXDES:
	case PDB_CMPDES:
	case PDB_AGRDES:
		pdb_sprintf(buffer, fmt, r->pdb.ftnote.num, r->pdb.ftnote.text);
		break;

	case PDB_HEADER:
		pdb_sprintf(buffer, fmt, r->pdb.header.class,
			r->pdb.header.date, r->pdb.header.type,
			r->pdb.header.id);
		break;

	case PDB_HELIX:
		pdb_sprintf(buffer, fmt, r->pdb.helix.serial_num,
			r->pdb.helix.id,
			r->pdb.helix.residues[0].name,
			r->pdb.helix.residues[0].chain_id,
			r->pdb.helix.residues[0].seq_num,
			r->pdb.helix.residues[0].insert_code,
			r->pdb.helix.residues[1].name,
			r->pdb.helix.residues[1].chain_id,
			r->pdb.helix.residues[1].seq_num,
			r->pdb.helix.residues[1].insert_code,
			r->pdb.helix.class, r->pdb.helix.comment);
		break;

	case PDB_HET:
		pdb_sprintf(buffer, fmt, r->pdb.het.het_grp.name,
			r->pdb.het.het_grp.chain_id, r->pdb.het.het_grp.seq_num,
			r->pdb.het.het_grp.insert_code, r->pdb.het.num_atoms,
			r->pdb.het.text);
		break;

	case PDB_MASTER:
		pdb_sprintf(buffer, fmt, r->pdb.master.num_remark,
			r->pdb.master.num_ftnote, r->pdb.master.num_het,
			r->pdb.master.num_helix, r->pdb.master.num_sheet,
			r->pdb.master.num_turn, r->pdb.master.num_site,
			r->pdb.master.num_transform,
			r->pdb.master.num_coordinate, r->pdb.master.num_ter,
			r->pdb.master.num_conect, r->pdb.master.num_seqres);
		break;

	case PDB_MODEL:
		pdb_sprintf(buffer, fmt, r->pdb.model.num);
		break;

	case PDB_MTRIX:
		pdb_sprintf(buffer, fmt, r->pdb.mtrix.row_num,
			r->pdb.mtrix.serial_num, r->pdb.mtrix.m1,
			r->pdb.mtrix.m2, r->pdb.mtrix.m3, r->pdb.mtrix.v,
			r->pdb.mtrix.given);
		break;

	case PDB_OBSLTE:
		pdb_sprintf(buffer, fmt, r->pdb.obslte.continuation,
			r->pdb.obslte.date, r->pdb.obslte.old_id,
			r->pdb.obslte.id_map[0], r->pdb.obslte.id_map[1],
			r->pdb.obslte.id_map[2], r->pdb.obslte.id_map[3],
			r->pdb.obslte.id_map[4], r->pdb.obslte.id_map[2],
			r->pdb.obslte.id_map[6], r->pdb.obslte.id_map[7]);
		break;

	case PDB_ORIGX:
		pdb_sprintf(buffer, fmt, r->pdb.origx.row_num, r->pdb.origx.o1,
			r->pdb.origx.o2, r->pdb.origx.o3, r->pdb.origx.t);
		break;

	case PDB_REVDAT:
		pdb_sprintf(buffer, fmt, r->pdb.revdat.modification,
			r->pdb.revdat.continuation, r->pdb.revdat.date,
			r->pdb.revdat.id, r->pdb.revdat.mod_type,
			r->pdb.revdat.corrections);
		break;

	case PDB_SCALE:
		pdb_sprintf(buffer, fmt, r->pdb.scale.row_num, r->pdb.scale.s1,
			r->pdb.scale.s2, r->pdb.scale.s3, r->pdb.scale.u);
		break;

	case PDB_SEQRES:
		pdb_sprintf(buffer, fmt, r->pdb.seqres.serial_num,
			r->pdb.seqres.chain_id, r->pdb.seqres.count,
			r->pdb.seqres.names[0], r->pdb.seqres.names[1],
			r->pdb.seqres.names[2], r->pdb.seqres.names[3],
			r->pdb.seqres.names[4], r->pdb.seqres.names[5],
			r->pdb.seqres.names[6], r->pdb.seqres.names[7],
			r->pdb.seqres.names[8], r->pdb.seqres.names[9],
			r->pdb.seqres.names[10], r->pdb.seqres.names[11],
			r->pdb.seqres.names[12]);
		break;

	case PDB_SHEET:
		sh = &r->pdb.sheet;
		shr0 = &sh->residues[0];
		shr1 = &sh->residues[1];
		sha0 = &sh->atoms[0].residue;
		sha1 = &sh->atoms[1].residue;
		pdb_sprintf(buffer, fmt, sh->strand_num,
			sh->id, sh->count,
			shr0->name, shr0->chain_id, shr0->seq_num,
			shr0->insert_code,
			shr1->name, shr1->chain_id, shr1->seq_num,
			shr1->insert_code,
			sh->sense,
			sh->atoms[0].name,
			sha0->name, sha0->chain_id, sha0->seq_num,
			sha0->insert_code,
			sh->atoms[1].name,
			sha1->name, sha1->chain_id, sha1->seq_num,
			sha1->insert_code);
		break;

	case PDB_SITE:
		shr0 = &r->pdb.site.residues[0];
		shr1 = &r->pdb.site.residues[1];
		sha0 = &r->pdb.site.residues[2];
		sha1 = &r->pdb.site.residues[3];
		pdb_sprintf(buffer, fmt, r->pdb.site.seq_num,
			r->pdb.site.id, r->pdb.site.count,
			shr0->name, shr0->chain_id, shr0->seq_num,
			shr0->insert_code,
			shr1->name, shr1->chain_id, shr1->seq_num,
			shr1->insert_code,
			sha0->name, sha0->chain_id, sha0->seq_num,
			sha0->insert_code,
			sha1->name, sha1->chain_id, sha1->seq_num,
			sha1->insert_code);
		break;

	case PDB_SPRSDE:
		pdb_sprintf(buffer, fmt, r->pdb.sprsde.continuation,
			r->pdb.sprsde.date, r->pdb.sprsde.id,
			r->pdb.sprsde.supersede[0], r->pdb.sprsde.supersede[1],
			r->pdb.sprsde.supersede[2], r->pdb.sprsde.supersede[3],
			r->pdb.sprsde.supersede[4], r->pdb.sprsde.supersede[5],
			r->pdb.sprsde.supersede[6], r->pdb.sprsde.supersede[7]);
		break;

	case PDB_SSBOND:
		pdb_sprintf(buffer, fmt, r->pdb.ssbond.seq_num,
			r->pdb.ssbond.residues[0].name,
			r->pdb.ssbond.residues[0].chain_id,
			r->pdb.ssbond.residues[0].seq_num,
			r->pdb.ssbond.residues[0].insert_code,
			r->pdb.ssbond.residues[1].name,
			r->pdb.ssbond.residues[1].chain_id,
			r->pdb.ssbond.residues[1].seq_num,
			r->pdb.ssbond.residues[1].insert_code,
			r->pdb.ssbond.comment);
		break;

	case PDB_SYMOP:
		pdb_sprintf(buffer, fmt, r->pdb.symop.row_num,
			r->pdb.symop.serial_num, r->pdb.symop.s1,
			r->pdb.symop.s2, r->pdb.symop.s3, r->pdb.symop.t);
		break;

	case PDB_TER:
		pdb_sprintf(buffer, fmt, r->pdb.ter.serial_num,
			r->pdb.ter.residue.name, r->pdb.ter.residue.chain_id,
			r->pdb.ter.residue.seq_num,
			r->pdb.ter.residue.insert_code);
		break;

	case PDB_TRNSFM:
		pdb_sprintf(buffer, fmt, r->pdb.trnsfm.result_serial_num,
			r->pdb.trnsfm.apply_serial_num,
			r->pdb.trnsfm.source_serial_num);
		break;

	case PDB_TURN:
		pdb_sprintf(buffer, fmt, r->pdb.turn.seq_num,
			r->pdb.turn.id,
			r->pdb.turn.residues[0].name,
			r->pdb.turn.residues[0].chain_id,
			r->pdb.turn.residues[0].seq_num,
			r->pdb.turn.residues[0].insert_code,
			r->pdb.turn.residues[1].name,
			r->pdb.turn.residues[1].chain_id,
			r->pdb.turn.residues[1].seq_num,
			r->pdb.turn.residues[1].insert_code,
			r->pdb.turn.comment);
		break;

	case PDB_TVECT:
		pdb_sprintf(buffer, fmt, r->pdb.tvect.serial_num,
			r->pdb.tvect.t1, r->pdb.tvect.t2, r->pdb.tvect.t3,
			r->pdb.tvect.comment);
		break;

	case PDB_USER:
		pdb_sprintf(buffer, fmt, r->pdb.user.subtype, r->pdb.user.text);
		break;

	case PDB_USER_PDBRUN:
		pdb_sprintf(buffer, fmt, r->pdb.user_pdbrun.version);
		pdb_pdbrun_version = r->pdb.user_pdbrun.version;
		break;

	case PDB_USER_EYEPOS:
		pdb_sprintf(buffer, fmt, r->pdb.user_eyepos.xyz[0],
			r->pdb.user_eyepos.xyz[1], r->pdb.user_eyepos.xyz[2]);
		break;

	case PDB_USER_ATPOS:
		pdb_sprintf(buffer, fmt, r->pdb.user_atpos.xyz[0],
			r->pdb.user_atpos.xyz[1], r->pdb.user_atpos.xyz[2]);
		break;

	case PDB_USER_WINDOW:
		pdb_sprintf(buffer, fmt, r->pdb.user_window.left,
			r->pdb.user_window.right, r->pdb.user_window.bottom,
			r->pdb.user_window.top, r->pdb.user_window.hither,
			r->pdb.user_window.yon);
		break;

	case PDB_USER_FOCUS:
		pdb_sprintf(buffer, fmt, r->pdb.user_focus.focus);
		break;

	case PDB_USER_VIEWPORT:
		pdb_sprintf(buffer, fmt, r->pdb.user_viewport.xmin,
			r->pdb.user_viewport.xmax, r->pdb.user_viewport.ymin,
			r->pdb.user_viewport.ymax);
		break;

	case PDB_USER_BGCOLOR:
		if (pdb_pdbrun_version < 6)
			sprintf(buffer, fmt, r->pdb.user_bgcolor.rgb[0],
				r->pdb.user_bgcolor.rgb[1],
				r->pdb.user_bgcolor.rgb[2]);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_bgcolor.rgb[0],
				r->pdb.user_bgcolor.rgb[1],
				r->pdb.user_bgcolor.rgb[2]);
		break;

	case PDB_USER_ANGLE:
		if (pdb_pdbrun_version < 6)
			pdb_sprintf(buffer, fmt, r->pdb.user_angle.which,
				r->pdb.user_angle.atom0,
				r->pdb.user_angle.atom1,
				r->pdb.user_angle.atom2,
				r->pdb.user_angle.atom3,
				r->pdb.user_angle.angle);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_angle.atom0,
				r->pdb.user_angle.atom1,
				r->pdb.user_angle.atom2,
				r->pdb.user_angle.atom3,
				r->pdb.user_angle.angle);
		break;

	case PDB_USER_DISTANCE:
		if (pdb_pdbrun_version < 6)
			pdb_sprintf(buffer, fmt, r->pdb.user_distance.which,
				r->pdb.user_distance.atom0,
				r->pdb.user_distance.atom1,
				r->pdb.user_distance.distance);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_distance.atom0,
				r->pdb.user_distance.atom1,
				r->pdb.user_distance.distance);
		break;

	case PDB_USER_FILE:
		if (pdb_pdbrun_version < 6)
			pdb_sprintf(buffer, fmt, r->pdb.user_file.filename);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_file.model,
						r->pdb.user_file.filename);
		break;

	case PDB_USER_MARKNAME:
		pdb_sprintf(buffer, fmt, r->pdb.user_markname.markname);
		break;

	case PDB_USER_MARK:
		pdb_sprintf(buffer, fmt, r->pdb.user_mark.markname);
		break;

	case PDB_USER_CNAME:
		if (pdb_pdbrun_version < 6)
			sprintf(buffer, fmt, r->pdb.user_cname.name,
				r->pdb.user_cname.rgb[0],
				r->pdb.user_cname.rgb[1],
				r->pdb.user_cname.rgb[2]);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_cname.rgb[0],
				r->pdb.user_cname.rgb[1],
				r->pdb.user_cname.rgb[2],
				r->pdb.user_cname.name);
		break;

	case PDB_USER_COLOR:
		if (pdb_pdbrun_version < 6)
			sprintf(buffer, fmt, r->pdb.user_color.spec,
				r->pdb.user_color.rgb[0],
				r->pdb.user_color.rgb[1],
				r->pdb.user_color.rgb[2]);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_color.rgb[0],
				r->pdb.user_color.rgb[1],
				r->pdb.user_color.rgb[2],
				r->pdb.user_color.spec);
		break;

	case PDB_USER_RADIUS:
		pdb_sprintf(buffer, fmt, r->pdb.user_radius.radius);
		break;

	case PDB_USER_OBJECT:
		pdb_sprintf(buffer, fmt, r->pdb.user_object.model);
		break;

	case PDB_USER_ENDOBJ:
		pdb_sprintf(buffer, fmt, r->pdb.user_endobj.model);
		break;

	case PDB_USER_CHAIN:
		if (pdb_pdbrun_version < 6)
			sprintf(buffer, fmt, r->pdb.user_chain.atom0,
				r->pdb.user_chain.atom1);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_chain.atom0,
				r->pdb.user_chain.atom1);
		break;

	case PDB_USER_GFX_BEGIN:
		if (r->pdb.user_gfx_begin.primitive == PDB_GFX_UNKNOWN)
			pdb_sprintf(buffer, fmt, r->pdb.user_gfx_begin.unknown);
		else
			pdb_sprintf(buffer, fmt, pdb_gfx_string(
					r->pdb.user_gfx_begin.primitive));
		break;

	case PDB_USER_GFX_END:
		pdb_sprintf(buffer, fmt);
		break;

	case PDB_USER_GFX_COLOR:
		if (pdb_pdbrun_version < 6)
			sprintf(buffer, fmt, r->pdb.user_gfx_color.spec,
				r->pdb.user_gfx_color.rgb[0],
				r->pdb.user_gfx_color.rgb[1],
				r->pdb.user_gfx_color.rgb[2]);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_gfx_color.rgb[0],
				r->pdb.user_gfx_color.rgb[1],
				r->pdb.user_gfx_color.rgb[2],
				r->pdb.user_gfx_color.spec);
		break;

	case PDB_USER_GFX_NORMAL:
		pdb_sprintf(buffer, fmt, r->pdb.user_gfx_normal.xyz[0],
			r->pdb.user_gfx_normal.xyz[1],
			r->pdb.user_gfx_normal.xyz[2]);
		break;

	case PDB_USER_GFX_VERTEX:
		pdb_sprintf(buffer, fmt, r->pdb.user_gfx_vertex.xyz[0],
			r->pdb.user_gfx_vertex.xyz[1],
			r->pdb.user_gfx_vertex.xyz[2]);
		break;

	case PDB_USER_GFX_FONT:
		if (pdb_pdbrun_version < 6)
			sprintf(buffer, fmt, r->pdb.user_gfx_font.name,
				r->pdb.user_gfx_font.size);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_gfx_font.size,
				r->pdb.user_gfx_font.name);
		break;

	case PDB_USER_GFX_TEXTPOS:
		pdb_sprintf(buffer, fmt, r->pdb.user_gfx_textpos.xyz[0],
			r->pdb.user_gfx_textpos.xyz[1],
			r->pdb.user_gfx_textpos.xyz[2]);
		break;

	case PDB_USER_GFX_LABEL:
		if (pdb_pdbrun_version < 6)
			sprintf(buffer, fmt, r->pdb.user_gfx_label.xyz[0],
				r->pdb.user_gfx_label.xyz[1],
				r->pdb.user_gfx_label.xyz[2],
				r->pdb.user_gfx_label.text);
		else
			pdb_sprintf(buffer, fmt, r->pdb.user_gfx_label.text);
		break;

	case PDB_USER_GFX_MOVE:
		pdb_sprintf(buffer, fmt, r->pdb.user_gfx_move.xyz[0],
			r->pdb.user_gfx_move.xyz[1],
			r->pdb.user_gfx_move.xyz[2]);
		break;

	case PDB_USER_GFX_DRAW:
		pdb_sprintf(buffer, fmt, r->pdb.user_gfx_draw.xyz[0],
			r->pdb.user_gfx_draw.xyz[1],
			r->pdb.user_gfx_draw.xyz[2]);
		break;

	case PDB_USER_GFX_MARKER:
		pdb_sprintf(buffer, fmt, r->pdb.user_gfx_marker.xyz[0],
			r->pdb.user_gfx_marker.xyz[1],
			r->pdb.user_gfx_marker.xyz[2]);
		break;

	case PDB_USER_GFX_POINT:
		pdb_sprintf(buffer, fmt, r->pdb.user_gfx_point.xyz[0],
			r->pdb.user_gfx_point.xyz[1],
			r->pdb.user_gfx_point.xyz[2]);
		break;

	default:
		(void) sprintf(buffer, "unknown pdb record #%d",
								r->record_type);
		break;
	}

	/* find last non-blank in buffer, and shorten it */
	t = NULL;
	for (s = buffer; *s != '\0'; s++)
		if (!isspace(*s))
			t = s + 1;
	if (t == NULL)		/* this should never happen, but ... */
		t = buffer;
	*t = '\0';
}

# ifdef vms
pdb_write_dummy()
{
	pdb_fmt_dummy();
}
# endif
