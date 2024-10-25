/*
 *	Copyright (c) 1993 The Regents of the University of California.
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
 *	$Id: read_format.i,v 1.2 93/04/21 19:55:49 gregc Exp $
 */
/* UNKNOWN */	NULL,
/* ANISOU */	"%6 %5d %4s%c%4s%c%4d%c %7d%7d%7d%7d%7d%7d",	/* SIGUIJ */
/* ATOM */	"%6 %5d %4s%c%4s%c%4d%c   %8f%8f%8f%6f%6f %3d",	/* HETATM, SIGATM */
/* AUTHOR */	"%9 %c%60s",		/* COMPND, EXPDTA, JRNL, SOURCE */
/* COMPND */	"%9 %c%60s",			/* AUTHOR */
/* CONECT */	"%6 %5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d",
/* CRYST1 */	"%6 %9f%9f%9f%7f%7f%7f %11s%4d",
/* END */	NULL,
/* FORMUL */	"%8 %2d  %4s%2d%c%51s",
/* FTNOTE */	"%7 %3d %59s",	/*  REMARK, SYMDES, MTXDES, CMPDES, AGRDES */
/* HEADER */	"%10 %40s%9s  %c%4s",
/* HELIX */	"%7 %3d %3s %4s%c %4d%c %4s%c %4d%c%2d%30s",
/* HET */	"%7 %4s %c%4d%c  %5d%5 %40s",
/* HETATM */	"%6 %5d %4s%c%4s%c%4d%c   %8f%8f%8f%6f%6f %3d",	/* ATOM */
/* JRNL */	"%9 %c%60s",			/* AUTHOR */
/* MASTER */	"%10 %5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d",
/* MTRIX */	"%5 %d %3d%10f%10f%10f%5 %10f   %2d",
/* OBSLTE */	"%8 %2d %9s %4s%6 %4s %4s %4s %4s %4s %4s %4s %4s",
/* ORIGX */	"%5 %d%4 %10f%10f%10f%5 %10f",	/* SCALE */
/* REMARK */	"%7 %3d %59s",			/* FTNOTE */
/* REVDAT */	"%7 %3d%2d %9s %7s %c%7 %31s",
/* SCALE */	"%5 %d%4 %10f%10f%10f%5 %10f",
/* SEQRES */	"%6 %4d %c %4d  %4s%4s%4s%4s%4s%4s%4s%4s%4s%4s%4s%4s%4s",
/* SHEET */	"%6 %4d %3s%2d %4s%c%4d%c %4s%c%4d%c%2d %4s%4s%c%4d%c %4s%4s%c%4d%c",
/* SIGATM */	"%6 %5d %4s%c%4s%c%4d%c   %8f%8f%8f%6f%6f %3d",	/* ATOM */
/* SIGUIJ */	"%6 %5d %4s%c%4s%c%4d%c %7d%7d%7d%7d%7d%7d",	/* ANISOU */
/* SITE */	"%7 %3d %3s %2d %4s%c%4d%c %4s%c%4d%c %4s%c%4d%c %4s%c%4d%c",
/* SOURCE */	"%9 %c%60s",			/* AUTHOR */
/* SPRSDE */	"%8 %2d %9s %4s%6 %4s %4s %4s %4s %4s %4s %4s %4s",
/* SSBOND */	"%7 %3d %4s%c %4d%c   %4s%c %4d%c%4 %30s",
/* TER */	"%6 %5d%6 %4s%c%4d%c",
/* TURN */	"%7 %3d %3s %4s%c%4d%c %4s%c%4d%c%4 %30s",
/* TVECT */	"%7 %3d%10f%10f%10f%30s",
/* USER */	"%4 %2s%66s",
/* MODEL */	"%9 %5d",
/* ENDMDL */	NULL,
/* EXPDTA */	"%9 %c%60s",			/* AUTHOR */
/* SYMDES */	"%7 %3d %59s",			/* FTNOTE */
/* SYMOP */	"%5 %d %3d%10f%10f%10f%5 %10f",
/* MTXDES */	"%7 %3d %59s",			/* FTNOTE */
/* CMPDES */	"%7 %3d %59s",			/* FTNOTE */
/* CMPONT */	"%7 %3d %4s%c %4d%c %4s%c %4d%c",
/* TRNSFM */	"%7 %3d %3d %3d",
/* AGRDES */	"%7 %3d %59s",			/* FTNOTE */
/* AGGRGT */	"%7 %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d",
