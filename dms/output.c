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
#include <math.h>
#include "dms_param.h"
#include "atoms.h"

#ifdef BILD_OUTPUT
#define	BYATOM	/* Color by atom ownership */
#undef	BYTYPE	/* Color by type of surface */
#endif
#undef	CHECKPT	/* Make sure points are okay */

static double	contact_area, reentrant_area;
static int	contact_points, reentrant_points;

void print_atom();

/*
 * print_ms:
 *	Print surface data in some format
 */
void print_ms(fp, alist, have_normal)
FILE	*fp;
ATOM	*alist;
int	have_normal;
{
	register ATOM	*ap;
#ifdef BILD_OUTPUT
	double		x, y, z;
#endif

#ifdef BILD_OUTPUT
	x = y = z = 0;
	for (ap = alist; ap != NULL; ap = ap->next) {
		x += (*app)->coord[0];
		y += (*app)->coord[1];
		z += (*app)->coord[2];
	}
	fprintf(fp, ".tran %.3f %.3f %.3f\n",
		-x / natom, -y / natom, -z / natom);
#endif
	for (ap = alist; ap != NULL; ap = ap->next)
		print_atom(fp, ap, have_normal, alist);

	/*
	 * Report statistics
	 */
	fprintf(stderr, "%d points\t(%d contact, %d reentrant)\n",
		contact_points + reentrant_points, contact_points,
		reentrant_points);
	fprintf(stderr, "%.2f sq. A\t(%.2f contact, %.2f reentrant)\n",
		contact_area + reentrant_area, contact_area,
		reentrant_area);
	fprintf(stderr, "%.2f pts/sq.A\t(%.2f contact, %.2f reentrant)\n",
		(contact_points + reentrant_points) /
		(contact_area + reentrant_area),
		contact_points / contact_area,
		reentrant_area == 0 ? 0.0 : reentrant_points / reentrant_area);
}

#ifndef BILD_OUTPUT
/* ARGSUSED */
#endif
/*
 * print_atom:
 *	Print the atom and its associated surface points
 */
void print_atom(fp, ap, have_normal, alist)
FILE	*fp;
ATOM	*ap;
int	have_normal;
ATOM	*alist;
{
	register SURFACE	*sp;
	register POINT		*pp, *endpp;
#ifndef BILD_OUTPUT
	register char		*typep;
	register POINT		*np;
#else
#ifdef BYATOM
	static int		color = 1;
#endif
#endif

#ifdef BILD_OUTPUT
#ifdef BYATOM
	fprintf(fp, ".color %d\n", color);
	color = color % 7 + 1;
#endif
	fprintf(fp, ".cmov %.3f %.3f %.3f\n%s\n", ap->coord[0], ap->coord[1],
		ap->coord[2], ap->atname);
#else
	fprintf(fp, "%3s %4s %4.4s%8.3f %8.3f %8.3f A\n", ap->restype,
		ap->resseq, ap->atname, ap->coord[0], ap->coord[1],
		ap->coord[2]);
#endif
	if (!ap->wanted)
		return;
	for (sp = ap->surface; sp != NULL; sp = sp->next) {
		if (sp->type == CONTACT) {
			contact_points += sp->npoint;
			contact_area += sp->npoint * sp->area;
		}
		else {
			reentrant_points += sp->npoint;
			reentrant_area += sp->npoint * sp->area;
		}
#ifndef BILD_OUTPUT
		switch (sp->type) {
		  case CONTACT:
			typep = "SC0";
			break;
		  case TREENTRANT:
			typep = "SS0";
			break;
		  case SREENTRANT:
			typep = "SR0";
			break;
		}
		np = sp->normal;
#else
#ifdef BYTYPE
		fprintf(fp, ".color %d\n", sp->type + 1);
#endif
#endif
		endpp = sp->position + sp->npoint;
		for (pp = sp->position; pp < endpp; pp++) {
#ifdef CHECKPT
			verify_point(pp, alist);
#endif
#ifdef BILD_OUTPUT
			fprintf(fp, ".dot %.3f %.3f %.3f\n", pp->coord[0],
				pp->coord[1], pp->coord[2]);
#else
			fprintf(fp, "%3s %4s %4.4s%8.3f %8.3f %8.3f %3s %6.3f",
				ap->restype, ap->resseq, ap->atname,
				pp->coord[0], pp->coord[1], pp->coord[2],
				typep, sp->area);
			if (have_normal) {
				fprintf(fp, " %6.3f %6.3f %6.3f", np->coord[0],
					np->coord[1], np->coord[2]);
				np++;
			}
			(void) putc('\n', fp);
#endif
		}
	}
}

#ifdef CHECKPT
/*
 * verify_point:
 *	Verify that the point is not inside any atom
 */
verify_point(pp, alist)
register POINT	*pp;
ATOM		*alist;
{
	register ATOM	*ap;
	register int	i;
	double		delta, distsq;

	for (ap = alist; ap != NULL; ap = ap->next) {
		distsq = 0;
		for (i = 0; i < 3; i++) {
			delta = ap->coord[i] - pp->coord[i];
			distsq += delta * delta;
		}
		if (distsq + 0.1 < ap->radius * ap->radius) {
			fprintf(stderr, "Point too close to %s@%s\n",
				ap->atname, ap->resseq);
			break;
		}
	}
}
#endif
