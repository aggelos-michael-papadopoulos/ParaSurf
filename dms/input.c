/*

Copyright (c) <2002> The Regents of the University of California.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions, and the following disclaimer.
  2. Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions, and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
  3. Redistributions must acknowledge that this software was
     originally developed by the UCSF Computer Graphics Laboratory
     under support by the NIH National Center for Research Resources,
     grant P41-RR01081.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <strings.h>
#include <pdb.h>
#include "dms_param.h"
#include "atoms.h"
#include "wanted.h"

#ifndef	DEBUG
#define	STATIC	static
#else
#define	STATIC
#endif

#ifndef	TRUE
#define	TRUE	1
#define	FALSE	0
#endif

STATIC int	is_aa();
void scopy(char*, char*, int);
char		(*resseqs)[RN_LEN];
int		numresseq;

/*
 * read_pdb:
 *	Read the given pdb file and put all the atoms into a list.
 */
ATOM *
read_pdb(file, all)
char	*file;
int	all;
{
	register ATOM		*cp;
	FILE			*fp;
	ATOM			*alist, *ap;
	pdb_record		r;
	struct pdb_atom		*atp;
	pdb_residue		*resp;
	extern char		*emalloc();
	char			prevresseq[RN_LEN];
	int			index;

	if ((fp = fopen(file, "r")) == NULL) {
		perror(file);
		return NULL;
	}
	alist = NULL;
	prevresseq[0] = '\0';
	numresseq = 0;
	do {
		r = pdb_read_record(fp);
		switch(r.record_type) {
		  case PDB_END:
		  case PDB_UNKNOWN:
		  case PDB_TER:
			break;
		  case PDB_HETATM:
			if (!all)
				break;
			/* FALLTHROUGH */
		  case PDB_ATOM:
			atp = &r.pdb.atom;
			resp = &atp->residue;
			if (!all && !is_aa(resp->name))
				break;
			cp = (ATOM *) emalloc(sizeof (ATOM));
			cp->coord[0] = r.pdb.atom.x;
			cp->coord[1] = r.pdb.atom.y;
			cp->coord[2] = r.pdb.atom.z;
			scopy(cp->atname, atp->name, sizeof (atp->name));
			scopy(cp->restype, resp->name, sizeof (resp->name));
			/*
			 * The residue sequence is composed of the
			 * sequence number concatenated with the insertion
			 * code and chain identifier (if they exist)
			 */
			(void) sprintf(cp->resseq, "%d", resp->seq_num);
			if (isalnum(resp->insert_code))
				(void) strncat(cp->resseq,
					&resp->insert_code, 1);
			if (isalnum(resp->chain_id))
				(void) strncat(cp->resseq,
					&resp->chain_id, 1);
			if (r.record_type == PDB_HETATM)
				(void) strncat(cp->resseq, "*", 1);
			if (strcmp(cp->resseq, prevresseq) != 0) {
				strcpy(prevresseq, cp->resseq);
				numresseq++;
			}
			cp->resindex = numresseq-1;
			cp->next = NULL;
			if (alist == NULL)
				alist = cp;
			else
				ap->next = cp;
			ap = cp;
			break;
		  default:
			break;
		}
	} while (r.record_type != PDB_END);
	(void) fclose(fp);
	if (alist == NULL)
		return alist;
	resseqs = (char (*)[RN_LEN]) emalloc(numresseq * RN_LEN);
	strcpy(resseqs[0], alist->resseq);
	index = 0;
	for (ap = alist->next; ap != NULL; ap = ap->next) {
		if (strcmp(ap->resseq, resseqs[index]) != 0) {
			index++;
			strcpy(resseqs[index], ap->resseq);
		}
	}
	if (index+1 != numresseq) {
		fprintf(stderr,
		  "Internal error: residue ID numbering mismatch.\n");
		exit(1);
	}
	return alist;
}

/*
 * scopy:
 *	Copy strings without copying blanks
 */
void scopy(to, from, max)
register char	*to, *from;
register int	max;
{
	while (max-- > 0 && *from != '\0')
		if (!isspace(*from))
			*to++ = *from++;
		else
			from++;
	*to = '\0';
}

/*
 * read_kraut:
 *	Read the given kraut format file into an atom list
 */
/* ARGSUSED */
ATOM *
read_kraut(file, all)
char	*file;
int	all;
{
	/*
	 * Not implemented yet (maybe never).  Just fake as if
	 * we had not seen any atoms.
	 */
	fputs("-k is unimplemented\n", stderr);
	return NULL;
}

/*
 * read_midas:
 *	Read the given midas format file into an atom list
 */
/* ARGSUSED */
ATOM *
read_midas(file, all)
char	*file;
int	all;
{
	/*
	 * Not implemented yet (maybe never).  Just fake as if
	 * we had not seen any atoms.
	 */
	fputs("-m is unimplemented\n", stderr);
	return NULL;
}

static char	*aa[] = {
	"  A", "  C", "  G", "  T", "  U",
	"ALA", "ARG", "ASN", "ASP", "CPR", "CYS", "CYX", "CYZ", "GLN",
	"GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
	"SER", "THR", "TRP", "TYR", "VAL",
	(char *) NULL
};

/*
 * is_aa:
 *	Decide whether the given residue type is an amino acid
 *	or nucleic acid.
 */
STATIC
int
is_aa(type)
char	*type;
{
	register int	n;
	register char	**aap;

	for (aap = aa; *aap != NULL; aap++) {
		n = strcmp(type, *aap);
		if (n == 0)
			return TRUE;
		else if (n < 0)
			break;
	}
	return FALSE;
}

/*
 * read_wanted:
 *	Read the "wanted list" of atoms from the given file
 */
WANTED *
read_wanted(fp)
FILE	*fp;
{
	register WANTED	*cp, *wp;
	char		buf[BUFSIZ], atname[AN_LEN];
	char		orig[BUFSIZ], orig2[BUFSIZ], *fields[4];
	int		numfields;
	extern char	*emalloc();
	static char	*errmsg = "Residues with chain IDs and/or insertion codes should have those appended\n\twith no spaces, e.g. residue 55 of chain E with insertion code A\n\twould be identified as 55AE.\n";
	static char	*errmsg2 = "Could not find residue '%s' in PDB file.\nThe residue was mentioned in site file line:\n%s\n";

	if (fp == NULL)
		return NULL;

	wp = NULL;
	while (fgets(buf, sizeof buf, fp) != NULL) {
		strcpy(orig, buf);
		if ((numfields = tokenize(buf, fields, 4)) != 3) {
			if (numfields == 0)
				/* don't penalize blank lines */
				continue;
			fputs("Illegal line in -i file:\n", stderr);
			fputs(orig, stderr);
			if (numfields > 3) {
				fputs(errmsg, stderr);
			}
			exit(1);
		}
		strcpy(atname, fields[2]);
		cp = (WANTED *) emalloc(sizeof (WANTED));
		cp->startres = locres(fields[1]);
		if (cp->startres < 0) {
			/* no such residue */
			fprintf(stderr, errmsg2, fields[1], orig);
			exit(1);
		}
		if (strcmp(atname, "*") == 0) {
			cp->type = W_ANY;
		}
		else if (strcmp(atname, "FRM") == 0) {
			cp->type = W_RANGE;
			if (fgets(buf, sizeof buf, fp) == NULL) {
				fputs("Unexpected EOF in -i file\n", stderr);
				(void) free((char *) cp);
				break;
			}
			strcpy(orig2, buf);
			numfields = tokenize(buf, fields, 4);
			if (numfields == 0
			  || strcmp(fields[numfields-1], "TO") != 0) {
				fputs("No TO after FRM in -i file\n", stderr);
				fprintf(stderr, "FRM line was: %s\n", orig);
				(void) free((char *) cp);
				exit(1);
			}
			if (numfields > 3) {
				fputs(errmsg, stderr);
				exit(1);
			}
			cp->w.endres = locres(fields[1]);
			if (cp->w.endres < 0) {
				fprintf(stderr, errmsg2, fields[1], orig2);
				exit(1);
			}
			if (cp->w.endres < cp->startres) {
				fprintf(stderr,
				  "End of FRM/TO range before begining!\n");
				fprintf(stderr, "FRM/TO lines were:%s%s\n",
				  orig, orig2);
				exit(1);
			}
		}
		else {
			cp->type = W_ATOM;
			(void) strcpy(cp->w.atom, atname);
		}
		cp->next = wp;
		wp = cp;
	}
	return wp;
}
