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
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/file.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "dms_param.h"
#include "atoms.h"
#include "protocol.h"

#ifndef	TRUE
#define	TRUE	1
#define	FALSE	0
#endif

#ifndef	PI
#define	PI	3.141592
#endif

#define	ROUND(x)	((int) ((x) + 0.5))
#define	ABS(x)		((x) < 0 ? -(x) : (x))

#define LOG		/* Log input and output */

#ifndef	DEBUG
#undef	LOG
#endif

static void	execute(char *cp);
static void	paramdata(double f1, double f2, int n);
static void	atomdata(int n);
static void	neighbordata(int n);
static void	probedata(int n);
static void	contacts(int n);
static void	s_probes(int n, int m);
static void	csurface(int n);
static void	tsurface(int n, int m);
static void	psurface(int n);

extern char	*sbrk(unsigned long);
extern char	*emalloc(unsigned int);

/*
 * SERVER:
 *	Compute server for MS.
 *	This server is started via inetd so the input comes from stdin
 *	and output goes to stdout.
 */
main()
{
	char		buf[BUFSIZ];
	int		lockfd;

	(void) chdir("/tmp");
#ifdef DEBUG
	(void) freopen("/tmp/dmsdlog", "w", stderr);
#ifdef BSD
	(void) setlinebuf(stderr);
#endif
	fprintf(stderr, "server started\n");
#endif

	/*
	 * Try locking the lock file.  If we fail, another dms
	 * server is already running so we just exit.  If the file
	 * doesn't exist, then we just assume that it's okay to run.
	 */
	lockfd = open(LOCK_FILE, 0);
	if (lockfd >= 0 && flock(lockfd, LOCK_EX) < 0) {
		printf("Go away, I'm busy.\n");
		exit(0);
	}

	/*
	 * Now we nice ourselves down to background priority so
	 * normal jobs will still get the CPU
	 */
#ifdef BSD
	(void) setpriority(PRIO_PROCESS, 0, NICE_PRIORITY);
#else
	(void) nice(NICE_PRIORITY);
#endif

	/*
	 * Read commands and execute them until we reach EOF
	 */
	while (fgets(buf, sizeof buf, stdin) != NULL) {
#ifdef LOG
		fprintf(stderr, "<-- %s", buf);
#endif
		execute(buf);
		(void) fflush(stdout);
	}

	/*
	 * Cleanup and exit
	 */
#ifdef DEBUG
	fprintf(stderr, "Memory high water mark: %d\n", sbrk(0));
#endif
	exit(0);
}

typedef struct	keyword_def	{
	char	*string;
	int	length;
	int	type;
}	KEYWORD;

#define	C_PARAM		0
#define	C_ATOM		1
#define	C_NEIGHBOR	2
#define	C_PROBEDATA	3
#define	C_CONTACT	4
#define	C_PROBE		5
#define	C_CSURF		6
#define	C_TSURF		7
#define	C_PSURF		8

static KEYWORD	keyword[] = {
	{	"paramdata",	9,	C_PARAM,	},
	{	"atomdata",	8,	C_ATOM,		},
	{	"neighbordata",	12,	C_NEIGHBOR,	},
	{	"probedata",	9,	C_PROBEDATA,	},
	{	"contacts",	8,	C_CONTACT,	},
	{	"probes",	6,	C_PROBE,	},
	{	"csurface",	8,	C_CSURF,	},
	{	"tsurface",	8,	C_TSURF,	},
	{	"psurface",	8,	C_PSURF,	},
};
#define	NKEY	(sizeof keyword / sizeof (KEYWORD))

/*
 * execute:
 *	Parse the given command and call the appropriate handler
 */
static
void
execute(char *cp)
{
	register KEYWORD	*kp, *endkp;
	int			n, m;
	double			f1, f2;

	endkp = &keyword[NKEY];
	for (kp = keyword; kp < endkp; kp++)
		if (strncmp(cp, kp->string, kp->length) == 0)
			break;
	if (kp >= endkp) {
		printf("Unknown command: %s", cp);
#ifdef DEBUG
		fprintf(stderr, "Unknown command: %s", cp);
#endif
		return;
	}
	switch (kp->type) {
	  case C_PARAM:
		if (sscanf(cp, PI_COMMAND, &f1, &f2, &n) != 3)
			goto bad;
		paramdata(f1, f2, n);
		break;
	  case C_ATOM:
		if (sscanf(cp, AI_COMMAND, &n) != 1)
			goto bad;
		atomdata(n);
		break;
	  case C_NEIGHBOR:
		if (sscanf(cp, NI_COMMAND, &n) != 1)
			goto bad;
		neighbordata(n);
		break;
	  case C_PROBEDATA:
		if (sscanf(cp, PRI_COMMAND, &n) != 1)
			goto bad;
		probedata(n);
		break;
	  case C_CONTACT:
		if (sscanf(cp, C_COMMAND, &n) != 1)
			goto bad;
		contacts(n);
		break;
	  case C_PROBE:
		if (sscanf(cp, P_COMMAND, &n, &m) != 2)
			goto bad;
		s_probes(n, m);
		break;
	  case C_CSURF:
		if (sscanf(cp, CS_COMMAND, &n) != 1)
			goto bad;
		csurface(n);
		break;
	  case C_TSURF:
		if (sscanf(cp, TS_COMMAND, &n, &m) != 2)
			goto bad;
		tsurface(n, m);
		break;
	  case C_PSURF:
		if (sscanf(cp, PS_COMMAND, &n) != 1)
			goto bad;
		psurface(n);
		break;
	}
	return;
bad:
	printf("Malformed command: %s", cp);
#ifdef DEBUG
	fprintf(stderr, "Malformed command: %s", cp);
#endif
	return;
}

/*
 * The remainder of this file consists of routines that correspond to
 * commands.  All necessary types and global variables are declared here.
 */

typedef struct satom_def	{
	struct satom_def	*next;		/* Pointer to next neighbor */
	ATOM			atom;		/* Atom information */
	int			nnb;		/* Number of neighbors */
	AINDEX			*nb;		/* Neighbor list */
	int			nprobe;		/* Number of probes */
	int			*probe;		/* Probe list */
	int			nused;
#ifdef SORTED_NB
	double			dist;
#endif
}	SATOM;

typedef struct sphere_def	{
	struct sphere_def	*next;		/* Pointer to next sphere */
	double			radius;		/* Radius */
	double			area;		/* Area per point */
	int			npoint;		/* Number of points */
	POINT			*point;		/* Point list */
}	SPHERE;

typedef struct snb_def	{
	double	*coord;				/* Coordinate of neighbor */
	double	dist;				/* Distance between atoms */
	double	clip;				/* Clipping distance */
}	SNB;

typedef struct sangle_def	{
	struct sangle_def	*next;		/* Pointer to next angle */
	double			angle;		/* Probe angle */
	int			start;		/* Starting or ending? */
	int			real;		/* This section truly exist */
}	SANGLE;

double	radius, density;	/* Probe and surface parameters */
double	arclength;		/* Approximate arc length between points */
double	root3;			/* Square root of 3 */
int	want_normal;		/* Whether we should report normals */
int	natom;			/* Number of atoms */
SATOM	*atom;			/* Atom list */
int	nprobe;			/* Number of probes */
PROBE	*probe;			/* Probe list */
double	maxrad;			/* Maximum atomic radius */
SPHERE	*sphere;		/* List of prototype spheres */
int	maxsurf;		/* Size of surface list */
POINT	*surface;		/* Surface points list */
POINT	*normal;		/* Normal list (same size as surface) */

static SATOM	*find_satom(double x);
static int	touches(SATOM *s1, SATOM *s2, double *dp);
static int	probe_position(SATOM *a0, SATOM *a1, SATOM *a2,
		double ppos[2][3], int check_only);
static int	hidden(SATOM *s0, SATOM *s1, SATOM *s2, double coord[3]);
static int	occlude(SATOM *sap, double pos[3]);
static SPHERE	*get_sphere(double r);
static int	clipped(double coord[3], SNB *nlist, SNB *endsnp);
static SANGLE	*compute_angles(SATOM *s1, SATOM *s2, double mat[4][4],
		int *na);
static void	draw_tsections(double invmat[4][4], double tradius,
		double clip[2], int n, int m, SANGLE *alist);
static int	torus_data(ATOM *a1, ATOM *a2, double tcenter[3],
		double *tradius, double clip[2]);
static SANGLE	*probe_angle(int pnum, double mat[4][4], int n, int m);
static SANGLE	*insert_angle(SANGLE *head, SANGLE *new);
static void	tsection(double invmat[4][4], double tradius, double clip[2],
		double start, double end, int n, int m, int check);
static void	send_points(POINT *plist, int np, AINDEX aindex, double area,
		int type, POINT *nlist);
static void	check_point(PROBE *prp, int at, int up, int side, int npoint,
		POINT point[], short keep[]);

/*
 * paramdata:
 *	Take the parameter data and put them in the right places
 */
static
void
paramdata(double f1, double f2, int n)
{
	radius = f1;
	density = f2 * 2.75;
	want_normal = n;
	root3 = sqrt(3.0);

	/*
	 * The approximation
	 *	arclength = sqrt( sqrt(3) / (3 * density) )
	 * came from Eric Pettersen who approximated a sphere with a
	 * plane (an infinite sphere, you know).
	 */
	arclength = 1.0 / sqrt(root3 * density);

	fputs(PI_RESPONSE, stdout);
#ifdef LOG
	fputs(PI_RESPONSE, stderr);
#endif
}

/*
 * atomdata:
 *	Read in atomic data
 */
static
void
atomdata(int n)
{
	register int		cc;
	register SATOM		*sap, *endsap;
	register struct iovec	*ip;
	struct iovec		*iov, *endip;

	/*
	 * Update global variables
	 */
	natom = n;
	atom = (SATOM *) emalloc(n * sizeof (SATOM));
#ifdef DEBUG
	fprintf(stderr, "Atom data = %d bytes\n", n * sizeof (SATOM));
#endif

	/*
	 * Read in the data
	 */
	iov = (struct iovec *) emalloc(n * sizeof (struct iovec));
	endip = iov + n;
	for (ip = iov, sap = atom; ip < endip; ip++, sap++) {
		ip->iov_base = (caddr_t) &sap->atom;
		ip->iov_len = sizeof (ATOM);
	}
	if ((cc = freadv(stdin, iov, n)) != n * sizeof (ATOM)) {
		printf("atomdata: got %d bytes instead of %d.\n",
			cc, n * sizeof (ATOM));
#ifdef DEBUG
		fprintf(stderr, "atomdata: got %d bytes instead of %d.\n",
			cc, n * sizeof (ATOM));
#endif
		return;
	}

	/*
	 * Initialize data regarding spheres
	 */
	maxrad = 0;
	endsap = atom + n;
	for (sap = atom; sap < endsap; sap++) {
		if (sap->atom.radius > maxrad)
			maxrad = sap->atom.radius;
		sap->nb = NULL;
	}

	/*
	 * Cleanup and let caller know we've read things okay
	 */
	(void) free((char *) iov);
	fputs(AI_RESPONSE, stdout);
#ifdef LOG
	fputs(AI_RESPONSE, stderr);
#endif
}

/*
 * neighbordata:
 *	Read in neighbor data
 */
static
void
neighbordata(int n)
{
	register int		cc, size, total;
	register SATOM		*sap, *endsap;
	register struct iovec	*ip;
	struct iovec		*iov;

	/*
	 * Check for consistency
	 */
	if (n != natom) {
		printf("neighbordata: ncount(%d) != acount(%d)\n", n, natom);
#ifdef DEBUG
		fprintf(stderr, "neighbordata: ncount(%d) != acount(%d)\n",
			n, natom);
#endif
		return;
	}
	endsap = atom + natom;

	/*
	 * Read in the neighbor count for each atom
	 */
	iov = (struct iovec *) emalloc(natom * sizeof (struct iovec));
	for (sap = atom, ip = iov; sap < endsap; sap++, ip++) {
		ip->iov_base = (caddr_t) &sap->nnb;
		ip->iov_len = sizeof (int);
	}
	if ((cc = freadv(stdin, iov, n)) != n * sizeof (int)) {
		printf("neighbordata: read %d bytes instead of %d.\n", cc,
			n * sizeof (int));
#ifdef DEBUG
		fprintf(stderr, "neighbordata: read %d bytes instead of %d.\n",
			cc, n * sizeof (int));
#endif
		return;
	}

	/*
	 * Read in the actual data for each atom
	 */
	total = 0;
	n = 0;
	ip = iov;
	for (sap = atom; sap < endsap; sap++) {
		if (sap->nnb <= 0)
			continue;
		size = sap->nnb * sizeof (AINDEX);
		if (sap->nb == NULL)
			sap->nb = (AINDEX *) emalloc(size);
		ip->iov_base = (caddr_t) sap->nb;
		ip->iov_len = size;
		total += size;
		n++;
		ip++;
	}
	if (n > 0 && (cc = freadv(stdin, iov, n)) != total) {
		printf("neighbordata: read %d bytes instead of %d.\n", cc,
			total);
#ifdef DEBUG
		fprintf(stderr, "neighbordata: read %d bytes instead of %d.\n",
			cc, total);
#endif
		return;
	}
#ifdef DEBUG
	fprintf(stderr, "Neighbor data = %d bytes\n", total + n * sizeof (int));
#endif

	/*
	 * Cleanup and let caller know we've read things okay
	 */
	(void) free((char *) iov);
	fputs(NI_RESPONSE, stdout);
#ifdef LOG
	fputs(NI_RESPONSE, stderr);
#endif
}

/*
 * probedata:
 *	Read in probe and update atom information regarding probes
 */
static
void
probedata(int n)
{
	register int		i;
	register SATOM		*sap, *endsap;
	register PROBE		*pp, *endpp;
	register int		cc, size;

	/*
	 * Update global variables
	 */
	nprobe = n;
	if (n <= 0) {
		fputs(PRI_RESPONSE, stdout);
#ifdef LOG
		fputs(PRI_RESPONSE, stderr);
#endif
		return;
	}
	size = n * sizeof (PROBE);
	probe = (PROBE *) emalloc(size);
#ifdef DEBUG
	fprintf(stderr, "Probe data = %d bytes\n", size);
#endif

	/*
	 * Read in the data
	 */
	if ((cc = fread((char *) probe, sizeof (PROBE), n, stdin)) != n) {
		printf("probedata: got %d probes instead of %d.\n", cc, n);
#ifdef DEBUG
		fprintf(stderr, "probedata: got %d bytes instead of %d.\n",
			cc, size);
#endif
		return;
	}

	/*
	 * Now we update the atom data
	 */
	endsap = atom + natom;
	for (sap = atom; sap < endsap; sap++)
		sap->nprobe = sap->nused = 0;
	endpp = probe + nprobe;
	for (pp = probe; pp < endpp; pp++)
		for (i = 0; i < 3; i++)
			atom[pp->atom[i]].nprobe++;
	for (sap = atom; sap < endsap; sap++)
		if (sap->nprobe > 0)
			sap->probe = (int *) emalloc(sap->nprobe *
				sizeof (int));
		else
			sap->probe = NULL;
	for (pp = probe; pp < endpp; pp++)
		for (i = 0; i < 3; i++) {
			sap = &atom[pp->atom[i]];
			sap->probe[sap->nused++] = pp - probe;
		}

	/*
	 * Cleanup and let caller know we've read things okay
	 */
	fputs(PRI_RESPONSE, stdout);
#ifdef LOG
	fputs(PRI_RESPONSE, stderr);
#endif
}

/*
 * contacts:
 *	Compute the contacts for atom `n'
 */
static
void
contacts(int n)
{
	register int	i, nn;
	register SATOM	*sp, *prevsp;
	register SATOM	*sap, *me, *nlist;
	register SATOM	*min, *max;
	AINDEX		*narray;
	double		distsq;

	/*
	 * Initialize local variables
	 */
	me = &atom[n];

	/*
	 * Locate set of atoms within `maxrad' of the given atom in
	 * the X direction.  This gives us a set of atoms which is
	 * guaranteed to include all neighbors but is still smaller
	 * than the set of ALL atoms.  There are more efficient ways
	 * to locate smaller sets but this one is easy.
	 */
	min = find_satom(me->atom.coord[0] - me->atom.radius
		- 2 * radius - maxrad);
	max = find_satom(me->atom.coord[0] + me->atom.radius
		+ 2 * radius + maxrad) + 1;

	/*
	 * Loop through the set of atoms and locate neighbors
	 */
	nlist = NULL;
	nn = 0;
	for (sap = min; sap < max; sap++) {
		if (sap == me || !touches(me, sap, &distsq))
			continue;
		nn++;
#ifdef SORTED_NB
		sap->dist = distsq;
		prevsp = NULL;
		for (sp = nlist; sp != NULL; sp = sp->next) {
			if (sp->dist > distsq)
				break;
			prevsp = sp;
		}
		if (prevsp == NULL) {
			sap->next = nlist;
			nlist = sap;
		}
		else {
			sap->next = prevsp->next;
			prevsp->next = sap;
		}
#else
		sap->next = nlist;
		nlist = sap;
#endif
	}

	/*
	 * Report the neighbor list
	 */
	printf(C_RESPONSE, n, nn);
#ifdef LOG
	fprintf(stderr, C_RESPONSE, n, nn);
#endif
	if (nn > 0) {
		narray = (AINDEX *) emalloc(nn * sizeof (AINDEX));
		for (sap = nlist, i = 0; sap != NULL; sap = sap->next, i++)
			narray[i] = sap - atom;
		(void) fwrite((char *) narray, sizeof (AINDEX), nn, stdout);
		(void) free((char *) narray);
	}
}

/*
 * find_satom:
 *	When given an X coordinate, find the first atom whose
 *	X coordinate is greater than or equal to the given one.
 *	Note that the atoms are sorted by X already so we can
 *	use a binary search.
 */
static
SATOM *
find_satom(double x)
{
	register int	hi, lo, mid;

	hi = natom;
	lo = 0;
	mid = (hi + lo) / 2;
	while (mid > lo) {
		if (atom[mid].atom.coord[0] < x)
			lo = mid;
		else
			hi = mid;
		mid = (hi + lo) / 2;
	}
	return &atom[mid];
}

/*
 * touches:
 *	When given two atoms, determine whether they are neighbors
 *	(i.e. a probe will NOT fit between them).
 */
static
int
touches(SATOM *s1, SATOM *s2, double *dp)
{
	register int	i;
	double		distsq, delta;
	double		maxdistsq;

	/*
	 * Compute the maximum distance between these two atoms and
	 * still remain neighbors
	 */
	delta = s1->atom.radius + s2->atom.radius + 2 * radius;
	maxdistsq = delta * delta;

	/*
	 * We determine the square of the distance between the two
	 * atoms and compare against `maxdistsq'.  Note that we
	 * compute the distance in the order Z, Y, X since we know
	 * that X is probably okay and Z and Y will give quicker
	 * failure if there is one.
	 */
	distsq = 0;
	for (i = 2; i >= 0; i--) {
		delta = s1->atom.coord[i] - s2->atom.coord[i];
		distsq += delta * delta;
		if (distsq >= maxdistsq)
			return FALSE;
	}
	*dp = distsq;
	return TRUE;
}

/*
 * s_probes:
 *	Compute the probes that are adjacent to both given atoms
 */
static
void
s_probes(int n, int m)
{
	register int	i, j, k;
	register SATOM	*s0, *s1, *s2;
	register PROBE	*plist, *pp;
	register int	np;
	double		ppos[2][3];
	int		wanted;

	/*
	 * Initialize local variables
	 */
	s0 = &atom[n];
	s1 = &atom[m];
	wanted = s0->atom.wanted || s1->atom.wanted;

	/*
	 * Locate probes that may be in contact with the given
	 * atoms.  We need not report any probe that involves any
	 * atom whose index is less than `m' because it will already
	 * have been computed
	 */
	plist = NULL;
	np = 0;
	for (i = 0; i < s0->nnb; i++)
		for (j = 0; j < s1->nnb; j++) {
			if (s0->nb[i] != s1->nb[j])
				continue;
			s2 = &atom[s0->nb[i]];
			if (!wanted && !s2->atom.wanted)
				continue;
			k = probe_position(s0, s1, s2, ppos, s0->nb[i] <= m);
			if (k < 0) {
				/*
				 * s2 is between s0 and s1.  Just report that
				 * n and m cannot be neighbors and quit
				 */
				printf(P_BADNB, n, m);
#ifdef LOG
				fprintf(stderr, P_BADNB, n, m);
#endif
				goto bad;
			}
			if (k == 0)
				break;
			for (k = 0; k < 2; k++) {
#ifdef KEEP_REAL
				if (hidden(s0, s1, s2, ppos[k]))
					continue;
#endif
				np++;
				pp = (PROBE *) emalloc(sizeof (PROBE));
				pp->next = plist;
				plist = pp;
				pp->coord[0] = ppos[k][0];
				pp->coord[1] = ppos[k][1];
				pp->coord[2] = ppos[k][2];
				pp->atom[0] = n;
				pp->atom[1] = m;
				pp->atom[2] = s0->nb[i];
#ifdef KEEP_REAL
				pp->real = TRUE;
#else
				pp->real = !hidden(s0, s1, s2, ppos[k]);
#endif
			}
			break;
		}

	/*
	 * Report data back to user
	 */
	printf(P_RESPONSE, np);
#ifdef LOG
	fprintf(stderr, P_RESPONSE, np);
#endif

	/*
	 * Send the data, clean up and go home
	 */
	for (pp = plist; pp != NULL; pp = plist) {
		(void) fwrite((char *) pp, sizeof (PROBE), 1, stdout);
		plist = pp->next;
		free((char *) pp);
	}
	return;

bad:
	/*
	 * We get here when we "know" that there can be no probes
	 * sitting against bot given atoms.  Just report a zero
	 * and release data structures.
	 */
	printf(P_RESPONSE, 0);
#ifdef LOG
	fprintf(stderr, P_RESPONSE, 0);
#endif
	for (pp = plist; pp != NULL; pp = plist) {
		plist = pp->next;
		free((char *) pp);
	}
}

/*
 * probe_position:
 *	When given three atoms, find the two possible probe positions.
 *	First we transform the coordinates such that a0 is at the origin,
 *	a1 is on the z axis, and a2 is in the y-z plane.  This arrangement
 *	constrains the two probe positions of have the same y and z coordinates
 *	and complementary x coordinates.
 *	The three equations constraining distances are:
 *		sq(x) + sq(y) + sq(z) = sq(R0 + Rp)
 *		sq(x) + sq(y) + sq(z - z1) = sq(R1 + Rp)
 *		sq(x) + sq(y - y2) + sq(z - z2) = sq(R2 + Rp)
 *	from which we can solve for z, then y in terms of z, then x in
 *	terms of y and z.
 *
 *	probe_position returns one of three values:
 *		 1 - where both probe positions have been found
 *		 0 - where no probe position is found but a0 and a1
 *		     can still be neighbors
 *		-1 - where a0 and a1 cannot be neighbors because a2 is
 *		     in the way
 */
static
int
probe_position(SATOM *a0, SATOM *a1, SATOM *a2, double ppos[2][3], int check_only)
{
	register int	i, j;
	double		mat[4][4], invmat[4][4];
	double		r0, r1, r2, dz1, dz2;
	double		z1, y2, z2, t;
	double		dy, dz;
	double		c[3];

	/*
	 * Convert to simple coordinate system
	 */
	viewat(mat, invmat, a0->atom.coord, a1->atom.coord, a2->atom.coord);
	z1 = a1->atom.coord[0] * mat[0][2] + a1->atom.coord[1] * mat[1][2] +
		a1->atom.coord[2] * mat[2][2] + mat[3][2];
	y2 = a2->atom.coord[0] * mat[0][1] + a2->atom.coord[1] * mat[1][1] +
		a2->atom.coord[2] * mat[2][1] + mat[3][1];
	z2 = a2->atom.coord[0] * mat[0][2] + a2->atom.coord[1] * mat[1][2] +
		a2->atom.coord[2] * mat[2][2] + mat[3][2];

	/*
	 * Solve in simple coordinate system.
	 */
	r0 = a0->atom.radius + radius;
	r1 = a1->atom.radius + radius;
	c[2] = (r0 * r0 - r1 * r1 + z1 * z1) / (2 * z1);
	r2 = a2->atom.radius + radius;
	dz1 = c[2] - z1;
	dz2 = c[2] - z2;
	c[1] = (r1 * r1 - r2 * r2 - dz1 * dz1 + dz2 * dz2 + y2 * y2) / (2 * y2);
	t = r0 * r0 - c[1] * c[1] - c[2] * c[2];

	/*
	 * Now check whether the probe position is in "real" space.  If
	 * not determine which atom is in between the others.
	 */
	if (t <= 0) {
		c[1] = -sqrt(r0 * r0 - c[2] * c[2]);
		dy = y2 - c[1];
		dz = z2 - c[2];
		if (dy * dy + dz * dz <= r2 * r2)
			return -1;
		return 0;
	}

	/*
	 * Probe positions are valid, convert back to original
	 * coordinate system and return
	 */
	if (check_only)
		return 0;
	c[0] = sqrt(t);
	for (i = 0; i < 3; i++) {
		ppos[0][i] = invmat[3][i];
		for (j = 0; j < 3; j++)
			ppos[0][i] += c[j] * invmat[j][i];
	}
	c[0] = -c[0];
	for (i = 0; i < 3; i++) {
		ppos[1][i] = invmat[3][i];
		for (j = 0; j < 3; j++)
			ppos[1][i] += c[j] * invmat[j][i];
	}
	return 1;
}

/*
 * hidden:
 *	Determine whether the given coordinates is occluded by
 *	any of the neighbors of the given atoms
 */
static
int
hidden(SATOM *s0, SATOM *s1, SATOM *s2, double coord[3])
{
	register int	i;

	/*
	 * First we unmark all the atoms that need work.
	 * Then we mark the three that, by definition, cannot occlude
	 * the probe position.
	 */
	for (i = 0; i < s0->nnb; i++)
		atom[s0->nb[i]].nused = 0;
	for (i = 0; i < s1->nnb; i++)
		atom[s1->nb[i]].nused = 0;
	for (i = 0; i < s2->nnb; i++)
		atom[s2->nb[i]].nused = 0;
	s0->nused = s1->nused = s2->nused = 1;

	/*
	 * Now we loop through all the atoms and check
	 */
	for (i = 0; i < s0->nnb; i++)
		if (atom[s0->nb[i]].nused++ == 0)
			if (occlude(&atom[s0->nb[i]], coord))
				return TRUE;
	for (i = 0; i < s1->nnb; i++)
		if (atom[s1->nb[i]].nused++ == 0)
			if (occlude(&atom[s1->nb[i]], coord))
				return TRUE;
	for (i = 0; i < s2->nnb; i++)
		if (atom[s2->nb[i]].nused++ == 0)
			if (occlude(&atom[s2->nb[i]], coord))
				return TRUE;

	/*
	 * Nobody blocked it.  I guess it's okay
	 */
	return FALSE;
}

/*
 * occlude:
 *	Determine whether the given atom precludes placing a sphere
 *	at the given position
 */
static
int
occlude(SATOM *sap, double pos[3])
{
	register int	i;
	double		delta, distsq;
#ifndef ONE_OCCLUDE
	double		minds;
#endif
#define	EPSILON	1e-12

#ifndef ONE_OCCLUDE
	delta = sap->atom.radius + radius;
	minds = delta * delta;
#endif
	distsq = EPSILON;
	for (i = 0; i < 3; i++) {
		delta = sap->atom.coord[i] - pos[i];
		distsq += delta * delta;
#ifndef ONE_OCCLUDE
		if (distsq > minds)
			return FALSE;
#endif
	}
#ifdef ONE_OCCLUDE
	delta = sap->atom.radius + radius;
	if (distsq > delta * delta)
		return FALSE;
#endif
	return TRUE;
}

/*
 * csurface:
 *	Compute the contact surface of the given atom
 */
static
void
csurface(int n)
{
	register POINT	*pp, *endpp;
	register SATOM	*sap, *me;
	register int	i, j;
	register SNB	*nlist, *snp, *endsnp;
	register SPHERE	*sp;
	register POINT	*savepp, *savenp;
	double		delta, distsq, len, t;
	double		r1p, r1psq, r2psq, r1sq;
	double		coord[3];

	/*
	 * Initialize local variables
	 */
	me = &atom[n];
	if (me->nnb > 0) {
		nlist = (SNB *) emalloc(me->nnb * sizeof (SNB));
		endsnp = nlist + me->nnb;
	}
	else
		nlist = NULL;

	/*
	 * Compute clipping distance for each neighbor
	 */
	r1p = me->atom.radius + radius;
	r1psq = r1p * r1p;
	r1sq = me->atom.radius * me->atom.radius;
	for (i = 0; i < me->nnb; i++) {
		sap = &atom[me->nb[i]];
		snp = &nlist[i];
		snp->coord = sap->atom.coord;
		distsq = 0;
		for (j = 0; j < 3; j++) {
			delta = me->atom.coord[j] - sap->atom.coord[j];
			distsq += delta * delta;
		}
		snp->dist = sqrt(distsq);
		delta = sap->atom.radius + radius;
		r2psq = delta * delta;
		len = (r1psq - r2psq + distsq) / (2 * snp->dist);
		t = len * me->atom.radius / r1p;
		snp->clip = r1sq + distsq - 2 * snp->dist * t;
	}

	/*
	 * Now get a sphere and find all points which are not clipped
	 * by any of the neighbors.
	 */
	sp = get_sphere(me->atom.radius);
	endpp = sp->point + sp->npoint;
	if (sp->npoint > maxsurf) {
		if (surface != NULL) {
			(void) free((char *) surface);
			if (want_normal)
				(void) free((char *) normal);
		}
		maxsurf = sp->npoint;
		surface = (POINT *) emalloc(maxsurf * sizeof (POINT));
		if (want_normal)
			normal = (POINT *) emalloc(maxsurf * sizeof (POINT));
	}
	savepp = surface;
	savenp = normal;
	for (pp = sp->point; pp < endpp; pp++) {
		for (i = 0; i < 3; i++)
			coord[i] = me->atom.coord[i] + pp->coord[i];
		if (nlist != NULL && clipped(coord, nlist, endsnp))
			continue;
		for (i = 0; i < 3; i++)
			savepp->coord[i] = coord[i];
		savepp++;
		if (want_normal) {
			for (i = 0; i < 3; i++)
				savenp->coord[i] = pp->coord[i] / sp->radius;
			savenp++;
		}
	}

	/*
	 * Now that we have the points, just send them back to caller
	 * and release the neighbor list
	 */
	i = savepp - surface;
	printf(SURFACE_RESPONSE, n, i, CONTACT, sp->area, want_normal);
#ifdef LOG
	fprintf(stderr, SURFACE_RESPONSE, n, i, CONTACT, sp->area, want_normal);
#endif
	if (i > 0) {
		(void) fwrite((char *) surface, sizeof (POINT), i, stdout);
		if (want_normal) {
			(void) fwrite((char *) normal, sizeof (POINT), i,
				stdout);
		}
	}
	fputs(SURFACE_END, stdout);
#ifdef LOG
	fputs(SURFACE_END, stderr);
#endif

	if (nlist != NULL)
		(void) free((char *) nlist);
}

/*
 * get_sphere:
 *	Return a prototype sphere of the given radius
 */
static
SPHERE *
get_sphere(double r)
{
	register SPHERE	*sp;
	register int	i, j;
	register POINT	*pp;
	register int	nlayer, npoint;
	double		dphi, dtheta;
	double		phi, theta, z, rsinphi;

	/*
	 * Look for one already on the list
	 */
	for (sp = sphere; sp != NULL; sp = sp->next)
		if (sp->radius == r)
			return sp;

	/*
	 * Okay, it doesn't exist yet.  Let's make one.
	 */
	sp = (SPHERE *) emalloc(sizeof (SPHERE));
	sp->next = sphere;
	sphere = sp;
	sp->radius = r;

	/*
	 * Easy part is done.  Now figure out how many points there
	 * should be on the sphere.
	 */
	dphi = arclength / r;
	nlayer = ROUND(PI / dphi) + 1;
	dphi = PI / nlayer;
	sp->npoint = 0;
	phi = 0;
	for (i = 0; i < nlayer; i++) {
		dtheta = (phi == 0) ? PI * 2 : arclength / (r * sin(phi));
		npoint = ROUND(PI * 2 / dtheta);
		if (npoint <= 0)
			npoint = 1;
		sp->npoint += npoint;
		phi += dphi;
	}
	sp->area = (4 * PI * r * r) / sp->npoint;

	/*
	 * Allocate space for the points and, possibly, angles
	 */
	sp->point = (POINT *) emalloc(sp->npoint * sizeof (POINT));

	/*
	 * Now we actually generate the points
	 */
	pp = sp->point;
	phi = 0;
	for (i = 0; i < nlayer; i++) {
		rsinphi = r * sin(phi);
		z = r * cos(phi);
		dtheta = (rsinphi == 0) ? PI * 2 : arclength / rsinphi;
		npoint = ROUND(PI * 2 / dtheta);
		if (npoint <= 0)
			npoint = 1;
		dtheta = PI * 2 / npoint;
		theta = (i % 2) ? 0 : PI;
		for (j = 0; j < npoint; j++) {
			pp->coord[0] = rsinphi * cos(theta);
			pp->coord[1] = rsinphi * sin(theta);
			pp->coord[2] = z;
			pp++;
			theta += dtheta;
			if (theta > PI * 2)
				theta -= PI * 2;
		}
		phi += dphi;
	}

	return sp;
}

/*
 * clipped:
 *	Determine if the given coordinate is clipped by any
 *	of the given neighbors
 */
static
int
clipped(double coord[3], SNB *nlist, SNB *endsnp)
{
	register SNB	*snp;
	register int	i;
	double		delta, distsq;

	for (snp = nlist; snp < endsnp; snp++) {
		distsq = 0;
		for (i = 0; i < 3; i++) {
			delta = snp->coord[i] - coord[i];
			distsq += delta * delta;
		}
		if (distsq < snp->clip)
			return TRUE;
	}
	return FALSE;
}

/*
 * tsurface:
 *	Compute torus surface for neighbor pair
 */
static
void
tsurface(int n, int m)
{
	register int	i, j;
	register SATOM	*s1, *s2;
	register SANGLE	*ap, *anext;
	register SANGLE	*alist;
	double		tcenter[3], tradius, clip[2], coord[3];
	double		mat[4][4], invmat[4][4];
	int		na;

	/*
	 * Initialize local variables
	 */
	s1 = &atom[n];
	s2 = &atom[m];
	if (!torus_data(&s1->atom, &s2->atom, tcenter, &tradius, clip)) {
		fputs(SURFACE_END, stdout);
#ifdef LOG
		fputs(SURFACE_END, stderr);
#endif
		return;
	}
	if (clip[0] > 0)
		lookat(mat, invmat, tcenter, s1->atom.coord);
	else {
		/* Note that we get the coordinates from s2 since clip[0]
		 * may be zero which would result in division by zero if
		 * we use coordinates from s1 */
		for (i = 0; i < 3; i++)
			coord[i] = tcenter[i] +
				(tcenter[i] - s2->atom.coord[i]);
		lookat(mat, invmat, tcenter, coord);
	}

	alist = compute_angles(s1, s2, mat, &na);
	if (alist == NULL) {
#ifdef KEEP_REAL
		na = 0;
		for (i = 0; i < s1->nnb; i++)
			for (j = 0; j < s2->nnb; j++) {
				if (s1->nb[i] != s2->nb[j])
					continue;
				na++;
				break;
			}
#endif
		if (na == 0) {
			tsection(invmat, tradius, clip, 0.0, PI * 2,
				n, m, TRUE);
		}
	}
	else
		draw_tsections(invmat, tradius, clip, n, m, alist);

	/*
	 * Release the angle list and return to caller
	 */
	for (ap = alist; ap != NULL; ap = anext) {
		anext = ap->next;
		(void) free((char *) ap);
	}
	fputs(SURFACE_END, stdout);
#ifdef LOG
	fputs(SURFACE_END, stderr);
#endif
}

/*
 * compute_angles:
 *	Loop through the probe list for each atom and find those
 *	probes which are on both list.
 */
#ifdef KEEP_REAL
/*ARGSUSED*/
#endif
static
SANGLE *
compute_angles(SATOM *s1, SATOM *s2, double mat[4][4], int *na)
{
	register int	i, j;
	register PROBE	*pp;
	register SANGLE	*alist, *ap;

	alist = NULL;
	j = s2 - atom;
#ifndef KEEP_REAL
	*na = 0;
#endif
	for (i = 0; i < s1->nprobe; i++) {
		pp = &probe[s1->probe[i]];
		if (pp->atom[0] != j && pp->atom[1] != j && pp->atom[2] != j)
			continue;
#ifndef KEEP_REAL
		(*na)++;
#endif
		ap = probe_angle(s1->probe[i], mat, s1 - atom, j);
		ap->real = pp->real;
		alist = insert_angle(alist, ap);
	}
	return alist;
}

/*
 * draw_tsections:
 *	Loop through the angle list and draw torus sections
 */
static
void
draw_tsections(double invmat[4][4], double tradius, double clip[2], int n, int m, SANGLE *alist)
{
	register SANGLE	*ap, *anext;

	for (ap = alist; ap != NULL; ap = ap->next) {
		if (!ap->start || !ap->real)
			continue;
		anext = (ap->next == NULL) ? alist : ap->next;
		tsection(invmat, tradius, clip, ap->angle,
			anext->angle, n, m, TRUE);
	}
}

/*
 * torus_data:
 *	Compute data regarding the torus (e.g. center, radius, etc.)
 */
static
int
torus_data(ATOM *a1, ATOM *a2, double tcenter[3], double *tradius, double clip[2])
{
	register int	i;
	double		distsq, dist;
	double		r1p, r2p;
	double		r1psq, r2psq, len;
	double		lsq;

	/*
	 * Compute distance between atoms
	 */
	distsq = 0;
	for (i = 0; i < 3; i++) {
		dist = a1->coord[i] - a2->coord[i];
		distsq += dist * dist;
	}
	dist = sqrt(distsq);

	/*
	 * Compute torus center and radius
	 */
	r1p = a1->radius + radius;
	r1psq = r1p * r1p;
	r2p = a2->radius + radius;
	r2psq = r2p * r2p;
	len = (r1psq - r2psq + distsq) / (2 * dist);
	lsq = r1psq - len * len;
	if (lsq <= 0)
		return 0;
	*tradius = sqrt(lsq);
	for (i = 0; i < 3; i++)
		tcenter[i] = a1->coord[i] +
			(a2->coord[i] - a1->coord[i]) * len / dist;

	/*
	 * Now compute the clipping distance for the torus
	 */
	clip[0] = len * (1.0 - a1->radius / r1p);
	clip[1] = (dist - len) * (1.0 - a2->radius / r2p);
	return 1;
}

/*
 * probe_angle:
 *	Determine the angle that the probe lies at
 */
static
SANGLE *
probe_angle(int pnum, double mat[4][4], int n, int m)
{
	register int	i, j;
	register PROBE	*pp;
	register SANGLE	*sp;
	register ATOM	*ap;
	double		angle;
	double		coord[3];

	/*
	 * Allocate space for angle structure
	 */
	sp = (SANGLE *) emalloc(sizeof (SANGLE));
	sp->next = NULL;
	pp = &probe[pnum];

	/*
	 * Compute the polar coordinates of the probe
	 */
	for (i = 0; i < 3; i++) {
		coord[i] = mat[3][i];
		for (j = 0; j < 3; j++)
			coord[i] += pp->coord[j] * mat[j][i];
	}
	sp->angle = atan2(coord[1], coord[0]);

	/*
	 * Compute the polar coordinates of the atom which
	 * is adjacent to both given atoms
	 */
	for (i = 0; i < 3; i++)
		if (pp->atom[i] != n && pp->atom[i] != m)
			break;
	ap = &atom[pp->atom[i]].atom;
	for (i = 0; i < 3; i++) {
		coord[i] = mat[3][i];
		for (j = 0; j < 3; j++)
			coord[i] += ap->coord[j] * mat[j][i];
	}
	angle = atan2(coord[1], coord[0]);

	/*
	 * Now determine whether the torus section should start or
	 * end here
	 */
	angle = angle - sp->angle;
	while (angle < 0)
		angle += PI * 2;
	sp->start = (angle >= PI);
	if (sp->angle < 0)
		sp->angle += PI * 2;

	return sp;
}

/*
 * insert_angle:
 *	Insert an angle into a list and return the head of the list.
 *	If there are two identical angles, we put the start before the end
 *	to eliminate the surface since this mostly occurs near rings.
 */
static
SANGLE *
insert_angle(SANGLE *head, SANGLE *new)
{
	register SANGLE	*ap;

	/*
	 * See if the new angle should be at the beginning
	 */
	if (head == NULL || new->angle < head->angle
	|| (new->angle == head->angle && !head->start)) {
		new->next = head;
		return new;
	}

	/*
	 * Find the angle whose successor is greater than the new angle.
	 * Add the angle after this one and return the original head
	 * of the list.
	 */
	for (ap = head; ap->next != NULL; ap = ap->next) {
		if (ap->next->angle > new->angle
		|| (ap->next->angle == new->angle && !ap->next->start))
			break;
	}
	new->next = ap->next;
	ap->next = new;
	return head;
}

/*
 * tsection:
 *	Generate a section of a torus
 *
 *	The area of described by an arc swept through an angle is:
 *		A = (integral) (integral) (r de) ds
 *	where
 *		 r = Rt - Rs cos a
 *		ds = Rs da
 *	so
 *		A = (i) (i) (Rt - Rs cos a) Rs da de
 *		  = e a Rt Rs - e Rs Rs (integral) cos a da
 *		  = e Rt a Rs - e Rs Rs sin a
 *	Symbols are:
 *		Rs = radius of arc
 *		Rt = arc displacement from origin
 *		 a = angle of arc
 *		 e = angle swept through
 */
static
void
tsection(double invmat[4][4], double tradius, double clip[2], double start, double end, int n, int m, int check)
{
	register int	i, j;
	register int	npoint;
	register POINT	*pp, *np;
	register int	na, ne;
	POINT		*plist, *nlist;
	double		area;
	double		mina, maxa;
	double		aincr, eincr;
	double		arange, erange;
	double		e, a, mida;
	double		nclip[2];
	int		eoffset;
	int		startm;
	double		coord[3], xy;
	double		cose, sine;
	double		normal[3];

	/*
	 * First see whether the torus is really a torus.
	 * If the probe extends below the center line between
	 * the two atoms, only the portion which is above the center
	 * line should be used for sweeping out the torus.
	 */
	if (check && tradius < radius) {
		xy = sqrt(radius * radius - tradius * tradius);
		clip[1] = -clip[1];
		if (clip[1] < -xy && -xy < clip[0]) {
			nclip[0] = -xy;
			nclip[1] = -clip[1];
			tsection(invmat, tradius, nclip, start, end,
				n, m, FALSE);
		}
		if (clip[1] < xy && xy < clip[0]) {
			nclip[0] = clip[0];
			nclip[1] = -xy;
			tsection(invmat, tradius, nclip, start, end,
				n, m, FALSE);
		}
		return;
	}

	/*
	 * Figure out basic parameters
	 */
	maxa = -acos(clip[0] / radius);
	mina = -acos(-clip[1] / radius);
	arange = maxa - mina;
	if (end < start)
		end += PI * 2;
	erange = end - start;
	area = erange * tradius * arange * radius -
		erange * radius * radius *
		(sin(maxa + PI / 2) + sin(-PI / 2  - mina));

	/*
	 * Now figure out the maximum number of points there might be
	 * and allocate space for them
	 */
	ne = ROUND(erange * tradius / arclength);
	if (ne <= 0)
		return;
	na = ROUND(arange * radius / (arclength * root3 / 2));
	if (na <= 0)
		return;
	npoint = na * ne;
	plist = (POINT *) emalloc(npoint * sizeof (POINT));
	if (want_normal)
		nlist = (POINT *) emalloc(npoint * sizeof (POINT));
	else
		nlist = NULL;

	/*
	 * Now loop through a and e and generate the points
	 */
	pp = plist;
	np = nlist;
	eoffset = 0;
	mida = (mina + maxa) / 2;
	aincr = arange / na;
	a = mina + aincr / 2;
	startm = 0;
	while (na-- > 0) {
		/*
		 * Compute things that only depend on a
		 */
		coord[2] = radius * cos(a);
		if (want_normal)
			normal[2] = -coord[2] / radius;
		xy = radius * sin(a) + tradius;

		/*
		 * Loop through e and generate points
		 */
		ne = ROUND(erange * xy / arclength);
		if (ne > 0) {
			eincr = erange / ne;
			e = start + eincr / 2 * (eoffset + 1);
			eoffset = 1 - eoffset;
		}
		while (ne-- > 0) {
#ifdef DEBUG
			if (pp >= plist + npoint) {
				fprintf(stderr, "Point buffer overflow\n");
				abort();
			}
#endif
			cose = cos(e);
			sine = sin(e);
			coord[0] = xy * cose;
			coord[1] = xy * sine;
			for (i = 0; i < 3; i++) {
				pp->coord[i] = invmat[3][i];
				for (j = 0; j < 3; j++)
					pp->coord[i] += coord[j] * invmat[j][i];
			}
			pp++;
			e += eincr;
			if (!want_normal)
				continue;
			normal[0] = (tradius * cose - coord[0]) / radius;
			normal[1] = (tradius * sine - coord[1]) / radius;
			for (i = 0; i < 3; i++) {
				np->coord[i] = 0;	/* Just rotation */
				for (j = 0; j < 3; j++)
					np->coord[i] += normal[j] *
						invmat[j][i];
			}
			np++;
		}

		/*
		 * Increment a.  If we pass the -PI/2 threshold, we will
		 * start generating points on atom m.
		 */
		a += aincr;
		if (a > mida && startm == 0)
			startm = pp - plist;
	}

	/*
	 * Send point data back to the caller and clean up
	 */
	npoint = pp - plist;
	if (npoint > 0) {
		area = area / npoint;
		send_points(plist, startm, m, area, TREENTRANT, nlist);
		send_points(plist + startm, npoint - startm, n,
			area, TREENTRANT, nlist + startm);
	}
	(void) free((char *) plist);
	if (nlist != NULL)
		(void) free((char *) nlist);
}

/*
 * send_points:
 *	Send points back to caller
 */
static
void
send_points(POINT *plist, int np, AINDEX aindex, double area, int type, POINT *nlist)
{
	if (np <= 0)
		return;
	printf(SURFACE_RESPONSE, aindex, np, type, area, want_normal);
#ifdef LOG
	fprintf(stderr, SURFACE_RESPONSE, aindex, np, type, area,
		want_normal);
#endif
	(void) fwrite((char *) plist, sizeof (POINT), np, stdout);
	if (want_normal)
		(void) fwrite((char *) nlist, sizeof (POINT), np, stdout);
}

/*
 * psurface:
 *	Generate surface associated with a probe
 */
static
void
psurface(int n)
{
	register int	i, j, k;
	register POINT	*pp;
	register PROBE	*prp;
	register SATOM	*sap;
	register SPHERE	*sp;
	int		which;
	double		minr, r, delta;
	POINT		*point, *plist[3], *normal, *nlist[3];
	short		*keep, np[3];

	/*
	 * Initialize local variables and allocate space
	 */
	prp = &probe[n];
	sp = get_sphere(radius);
	for (i = 0; i < 3; i++) {
		np[i] = 0;
		plist[i] = (POINT *) emalloc(sp->npoint * sizeof (POINT));
	}
	point = (POINT *) emalloc(sp->npoint * sizeof (POINT));
	keep = (short *) emalloc(sp->npoint * sizeof (short));
	for (i = 0; i < sp->npoint; i++) {
		keep[i] = TRUE;
		pp = &point[i];
		for (j = 0; j < 3; j++)
			pp->coord[j] = sp->point[i].coord[j] + prp->coord[j];
	}
	if (want_normal) {
		normal = (POINT *) emalloc(sp->npoint * sizeof (POINT));
		for (i = 0; i < sp->npoint; i++) {
			pp = &normal[i];
			for (j = 0; j < 3; j++)
				pp->coord[j] = -sp->point[i].coord[j] / radius;
		}
		for (i = 0; i < 3; i++)
			nlist[i] = (POINT *) emalloc(sp->npoint *
				sizeof (POINT));
	}
	else {
		normal = NULL;
		nlist[0] = nlist[1] = nlist[2] = NULL;
	}

	/*
	 * Now we eliminate those points which do not fall within the
	 * spherical triangle formed by the three specified atoms.
	 * The way we do this is by defining a plane for each pair of
	 * atoms and the probe center.  The plane divides space into two
	 * half-spaces, one of which contains the third atom not used in
	 * defining the plane.  If a point does not fall in the same
	 * half-space as the third atom, then it is eliminated.  Only
	 * the wanted points will remain after all three planes have been
	 * used.
	 */
	for (i = 0; i < 3; i++)
		for (j = i + 1; j < 3; j++)
			check_point(prp, i, j, 3 - i - j,
				sp->npoint, point, keep);

	/*
	 * Now we divide the points up among the three atoms.
	 */
	for (i = 0; i < sp->npoint; i++) {
		if (!keep[i])
			continue;
		pp = &point[i];
		which = -1;
#ifdef lint
		minr = 0;
#endif
		for (j = 0; j < 3; j++) {
			sap = &atom[prp->atom[j]];
			r = 0;
			for (k = 0; k < 3; k++) {
				delta = pp->coord[k] - sap->atom.coord[k];
				r += delta * delta;
			}
			if (which == -1 || r < minr) {
				minr = r;
				which = j;
			}
		}
		if (want_normal)
			nlist[which][np[which]] = normal[i];
		plist[which][np[which]++] = *pp;
	}

	/*
	 * Send the points back to caller and clean up
	 */
	for (i = 0; i < 3; i++) {
		send_points(plist[i], np[i], prp->atom[i], sp->area,
			SREENTRANT, nlist[i]);
		(void) free((char *) plist[i]);
		if (want_normal)
			(void) free((char *) nlist[i]);
	}
	fputs(SURFACE_END, stdout);
#ifdef LOG
	fputs(SURFACE_END, stderr);
#endif
	(void) free((char *) point);
	(void) free((char *) keep);
	if (want_normal)
		(void) free((char *) normal);
}

/*
 * check_point:
 *	Determine which of the given points should be kept
 */
static
void
check_point(PROBE *prp, int at, int up, int side, int npoint, POINT point[], short keep[])
{
	register int	i, sign;
	double		mat[4][4], invmat[4][4];
	double		x;
	double		*dp;
	double		*fp;

	/*
	 * Construct a matrix which transforms the probe center to
	 * the origin, the at atom to the z-axis, and the up atom
	 * into the y-z plane.  Then transform the side atom to determine
	 * the sign of the x-component of the half-space to keep.
	 */
	viewat(mat, invmat, prp->coord, atom[prp->atom[at]].atom.coord,
		atom[prp->atom[up]].atom.coord);
	dp = atom[prp->atom[side]].atom.coord;
	x = dp[0] * mat[0][0] + dp[1] * mat[1][0] + dp[2] * mat[2][0]
		+ mat[3][0];
	sign = (x < 0) ? -1 : 1;

	/*
	 * Now we loop through the points and check each one
	 */
	for (i = 0; i < npoint; i++) {
		if (!keep[i])
			continue;
		fp = point[i].coord;
		x = fp[0] * mat[0][0] + fp[1] * mat[1][0] + fp[2] * mat[2][0]
			+ mat[3][0];
		if ((sign < 0 && x > 0) || (sign > 0 && x < 0))
			keep[i] = FALSE;
	}
}
