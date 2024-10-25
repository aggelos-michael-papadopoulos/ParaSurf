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
#ifndef	__ATOM__
#define	__ATOM__

#define	TREENTRANT	0
#define	SREENTRANT	1
#define	CONTACT		2

typedef struct point_def	{
	double	coord[3];
}	POINT;

typedef struct surface_def	{
	struct surface_def	*next;
	int			npoint;
	POINT			*position;
	POINT			*normal;
	double			area;
	int			type;
}	SURFACE;

#define	AN_LEN	6
#define	RN_LEN	9
#define	RT_LEN	6

typedef struct atom_def	{
	struct atom_def	*next;
	double		coord[3];
	char		atname[AN_LEN];
	char		resseq[RN_LEN];
	char		restype[RT_LEN];
	double		radius;
	int		wanted, resindex;
	SURFACE		*surface;
}	ATOM;

typedef int	AINDEX;

typedef struct neighbor_def	{
	int	nneighbor;
	AINDEX	*neighbor;
}	NEIGHBOR;

typedef struct probe_def	{
	struct probe_def	*next;
	AINDEX			atom[3];
	double			coord[3];
	int			real;
}	PROBE;

#define	ATOM_IS(ap,aname,rname)	\
	(strcmp((ap)->atname, aname) == 0 && strcmp((ap)->resseq, rname) == 0)

#endif
