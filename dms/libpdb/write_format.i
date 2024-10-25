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
 *	$Id: write_format.i,v 1.3 93/06/24 11:53:47 gregc Exp $
 */
"UNKNOWN:  ??%-6.6s??",
"ANISOU%5d %-4s%c%-4s%c%4d%c %7d%7d%7d%7d%7d%7d",	/* SIGUIJ */
"ATOM  %5d %-4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f %3D", /* HETATM, SIGATM */
"AUTHOR   %c%-60s",			/* COMPND, EXPDTA, JRNL, SOURCE */
"COMPND   %c%-60s",					/* AUTHOR */
"CONECT%5d%5D%5D%5D%5D%5D%5D%5D%5D%5D%5D",
"CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %-11s%4d",
"END",
"FORMUL  %2D  %-4s%2D%c%-51s",
"FTNOTE %3D %-59s",					/* REMARK */
"HEADER    %-40s%-11s%c%-4s",
"HELIX  %3D %3s %-4s%c %4d%c %-4s%c %4d%c%2D%-30s",
"HET    %-4s %c%4d%c  %5d     %-40s",
"HETATM%5d %-4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f %3D",
"JRNL     %c%-60s",					/* AUTHOR */
"MASTER    %5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d",
"MTRIX%1d %3d%10.6f%10.6f%10.6f     %10.5f   %2D",
"OBSLTE  %2D %-9s %-10s%-5s%-5s%-5s%-5s%-5s%-5s%-5s%-4s",
"ORIGX%1d    %10.6f%10.6f%10.6f     %10.5f",		/* SCALE */
"REMARK %3D %-59s",
"REVDAT %3D%2D %-9s %-7s %c       %-31s",
"SCALE%1d    %10.6f%10.6f%10.6f     %10.5f",		/* ORIGX */
"SEQRES%4d %c %4d  %-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s",
"SHEET %4D %3s%2d %-4s%c%4d%c %-4s%c%4d%c%2d %-4s%-4s%c%4D%c %-4s%-4s%c%4D%c",
"SIGATM%5d %-4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f %3D",
"SIGUIJ%5d %-4s%c%-4s%c%4d%c %7D%7D%7D%7D%7D%7D",	/* ANISOU */
"SITE   %3d %3s %2d %-4s%c%4D%c %-4s%c%4D%c %-4s%c%4D%c %-4s%c%4D%c",
"SOURCE   %c%-60s",					/* AUTHOR */
"SPRSDE  %2D %-9s %-10s%-5s%-5s%-5s%-5s%-5s%-5s%-5s%-4s",
"SSBOND %3D %-4s%c %4d%c   %-4s%c %4D%c    %-30s",
"TER   %5d      %-4s%c%4d%c",
"TURN   %3D %3s %-4s%c%4d%c %-4s%c%4d%c    %-30s",
"TVECT  %3D%10.5f%10.5f%10.5f%-30s",
"USER%-2s%-66s",
"MODEL    %5d",
"ENDMDL",
"EXPDTA   %c%-60s",					/* AUTHOR */
"SYMDES %3d %59s",					/* FTNOTE */
"SYMOP%1d %3d%10.6f%10.6f%10.6f     %10.5f",
"MTXDES %3d %59s",					/* FTNOTE */
"CMPDES %3d %59s",					/* FTNOTE */
"CMPONT %3d %4s%c %4d%c %4s%c %4d%c",
"TRNSFM %3d %3d %3d",
"AGRDES %3d %59s",					/* FTNOTE */
"AGGRGT %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d",
