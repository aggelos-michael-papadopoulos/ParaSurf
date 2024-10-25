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
#include "dms_param.h"
#include "wanted.h"
#include "atoms.h"

#define	DEF_DENSITY	1.0		/* Default surface density */
#define	MIN_DENSITY	0.1
#define	MAX_DENSITY	10.0
#define	DEF_PROBE	1.4		/* Default probe radius */
#define	MIN_PROBE	1.0
#define	MAX_PROBE	201.0

#define	MODE_PDB	0
#define	MODE_MIDAS	1
#define	MODE_KRAUT	2

#ifndef	TRUE
#define	TRUE	1
#define	FALSE	0
#endif

void usage(char *);

/*
 * MS:
 *	Compute the solvent accessible surface of a molecule
 *	as defined by Richards.
 */
main(ac, av)
int	ac;
char	**av;
{
	register int	c;
	register ATOM	*ap;
	int		input_mode;
	int		aflg, nflg;
	int		natom;
	int		verbose;
	int		fd;
	int		any_wanted;
	double		density, probe;
	char		*in_file, *out_file;
	FILE		*out_fp;
	FILE		*wanted_fp;
	WANTED		*wanted_list;
	ATOM		*atom_list, **atom_array;
	extern int	optind;
	extern char	*optarg;
	extern double	atof();
	WANTED		*read_wanted();
	ATOM		*read_pdb(), *read_midas(), *read_kraut();
	ATOM		**make_array();

	/*
	 * Initialize variables to default values which may be
	 * overridden by command line arguments
	 */
	verbose = FALSE;
	aflg = FALSE;
	nflg = FALSE;
	density = DEF_DENSITY;
	probe = DEF_PROBE;
	input_mode = MODE_PDB;
	out_fp = NULL;
	wanted_fp = NULL;

	/*
	 * We process the arguments in a strange way because
	 * we want to be backwards compatible with the old ms
	 * program
	 */
	if (ac < 4) {
		usage(av[0]);
		exit(1);
	}
	in_file = av[1];
	av[1] = av[0];
	ac--;
	av++;
	while ((c = getopt(ac, av, "ad:e:g:i:kpmnr:w:o:v")) != EOF)
		switch (c) {
		  case 'a':
			aflg = TRUE;
			break;
		  case 'd':
			density = atof(optarg);
			if (density < MIN_DENSITY || density > MAX_DENSITY) {
				fprintf(stderr,
				"Density must be between %.2f and %.2f\n",
				MIN_DENSITY, MAX_DENSITY);
				exit(1);
			}
			break;
		  case 'g':
			(void) fflush(stderr);
			if ((fd = creat(optarg, 0666)) < 0) {
				perror(optarg);
				exit(1);
			}
			(void) dup2(fd, 2);
			(void) close(fd);
			break;
		  case 'i':
			if ((wanted_fp = fopen(optarg, "r")) == NULL) {
				perror(optarg);
				exit(1);
			}
			break;
		  case 'p':
			input_mode = MODE_PDB;
			break;
		  case 'w':
			probe = atof(optarg);
			if (probe < MIN_PROBE || probe > MAX_PROBE) {
				fprintf(stderr,
				"Probe size must be between %.2f and %.2f\n",
				MIN_PROBE, MAX_PROBE);
				exit(1);
			}
			break;
		  case 'n':
			nflg = TRUE;
			break;
		  case 'o':
			if ((out_fp = fopen(optarg, "w")) == NULL) {
				perror(optarg);
				exit(1);
			}
			out_file = optarg;
			break;
		  case 'k':
			input_mode = MODE_KRAUT;
			break;
		  case 'm':
			input_mode = MODE_MIDAS;
			break;
		  case 'e':
		  case 'r':
			fprintf(stderr, "-%c is not implemented\n", c);
			exit(1);
		  case 'v':
			verbose = TRUE;
			break;
		  default:
			fprintf(stderr, "-%c is not recognized\n", c);
			usage(av[0]);
			exit(1);
		}
	if (out_fp == NULL) {
		fprintf(stderr, "missing -o option\n");
		usage(av[0]);
		exit(1);
	}
#ifdef BSD
	(void) setlinebuf(stderr);
#endif

	/*
	 * Read in the atoms
	 */
	switch (input_mode) {
	  case MODE_PDB:
		atom_list = read_pdb(in_file, aflg);
		break;
	  case MODE_KRAUT:
		atom_list = read_kraut(in_file, aflg);
		break;
	  case MODE_MIDAS:
		atom_list = read_midas(in_file, aflg);
		break;
	}
	if (atom_list == NULL) {
		fprintf(stderr, "No atoms in input file!\n");
		(void) fclose(out_fp);
		(void) unlink(out_file);
		exit(1);
	}

	/*
	 * Read in the "ignore" file so we know what constraints
	 * we have on input, and check them against atoms present
	 */
	wanted_list = read_wanted(wanted_fp);
	if (wanted_fp != NULL)
		(void) fclose(wanted_fp);

	any_wanted = 0;
	for (ap = atom_list; ap != NULL; ap = ap->next) {
		ap->wanted = (wanted_list == NULL || wanted(ap, wanted_list));
		if (ap->wanted)
			any_wanted = 1;
	}
	if (any_wanted == 0) {
		fprintf(stderr, "No atoms selected by site file.\n");
		(void) fclose(out_fp);
		(void) unlink(out_file);
		exit(1);
	}
	atom_array = make_array(atom_list, &natom);
	fprintf(stderr, "%d atoms\n", natom);

	/*
	 * Compute the surface
	 */
	compute_ms(atom_array, natom, probe, density, nflg, verbose);

	/*
	 * Output the surface in some appropriate format
	 */
	print_ms(out_fp, atom_list, nflg);
	(void) fclose(out_fp);

	/*
	 * All done.  Clean up.
	 */
	exit(0);
}

/*
 * usage:
 *	Print a usage message and a description of each argument
 */
void usage(prog_name)
char	*prog_name;
{
#if 0
	fprintf(stderr, "Usage: %s input_file [-a] [-d density] [-e file] [-g file] [-i file] [-k] [-m] [-p] [-n] [-r b-e] [-w radius] -o file\n", prog_name);
#else
	fprintf(stderr, "Usage: %s input_file [-a] [-d density] [-g file] [-i file] [-n] [-w radius] [-v] -o file\n", prog_name);
#endif
	fputs("\t-a\tuse all atoms, not just amino acids\n", stderr);
	fputs("\t-d\tchange density of points\n", stderr);
	fputs("\t-g\tsend messages to file\n", stderr);
	fputs("\t-i\tcalculate only surface for specified atoms\n", stderr);
#if 0
	fputs("\t-e\tcalculate only surface within ellipsoid\n", stderr);
	fputs("\t-k\tinput file is in Kraut format\n", stderr);
	fputs("\t-m\tinput file is in Midas format\n", stderr);
	fputs("\t-p\tinput file is in Protein Data Bank format\n", stderr);
	fputs("\t-r\tuse only specified residues\n", stderr);
#endif
	fputs("\t-n\tcalculate normals for surface points\n", stderr);
	fputs("\t-w\tchange probe radius\n", stderr);
	fputs("\t-v\tverbose\n", stderr);
	fputs("\t-o\tspecify output file name (required)\n", stderr);
}

/*
 * make_array:
 *	Create an array which points at elements of a list
 */
ATOM **
make_array(alist, nel)
ATOM	*alist;
int	*nel;
{
	register ATOM	*ap, **app, **aarray;
	register int	n;
	extern char	*emalloc();

	n = 0;
	for (ap = alist; ap != NULL; ap = ap->next)
		n++;
	aarray = (ATOM **) emalloc(n * sizeof (ATOM *));
	app = aarray;
	for (ap = alist; ap != NULL; ap = ap->next)
		*app++ = ap;
	*nel = n;
	return aarray;
}
