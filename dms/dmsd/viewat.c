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

static void	nmcrosprod();

void viewat(M, invM, P1, P2, P3)
double	M[4][4], invM[4][4];
double	P1[3], P2[3], P3[3];
{
	double	d12;
	double	P12X, P12Y, P12Z, P13X, P13Y, P13Z;
	double	lm[3][3];
	register int	i, j;

	P12X = P2[0] - P1[0];
	P12Y = P2[1] - P1[1];
	P12Z = P2[2] - P1[2];
	P13X = P3[0] - P1[0];
	P13Y = P3[1] - P1[1];
	P13Z = P3[2] - P1[2];

	d12 = sqrt((P12X)*(P12X) + (P12Y)*(P12Y) + (P12Z)*(P12Z));
	lm[0][2] = (P12X) / d12;
	lm[1][2] = (P12Y) / d12;
	lm[2][2] = (P12Z) / d12;
	nmcrosprod(P13X, P13Y, P13Z, P12X, P12Y, P12Z,
		&lm[0][0], &lm[1][0], &lm[2][0]);
	nmcrosprod(lm[0][2], lm[1][2], lm[2][2],
		lm[0][0], lm[1][0], lm[2][0], 
		&lm[0][1], &lm[1][1], &lm[2][1]);

	for (i = 0; i < 3; i++) {
		invM[3][i] = P1[i];
		M[i][3] = invM[i][3] = 0;
		for (j = 0; j < 3; j++) {
			M[i][j] = lm[i][j];
			invM[j][i] = M[i][j];
		}
	}
	M[3][0] = -P1[0]*M[0][0] - P1[1]*M[1][0] - P1[2]*M[2][0];
	M[3][1] = -P1[0]*M[0][1] - P1[1]*M[1][1] - P1[2]*M[2][1];
	M[3][2] = -P1[0]*M[0][2] - P1[1]*M[1][2] - P1[2]*M[2][2];
	M[3][3] = invM[3][3] = 1;
}

static
void
nmcrosprod(x1, y1, z1, x2, y2, z2, x3, y3, z3)
double	x1, y1, z1, x2, y2, z2;			/* r1 cross r2 */
double	*x3, *y3, *z3;				/* Normalized crossproduct */
{
	double	dis;	/* length of crossproduct vector r1 x r2 */
	double	x,y,z;

	x = y1*z2 - y2*z1;
	y = z1*x2 - z2*x1;
	z = x1*y2 - x2*y1;

	dis = sqrt((x*x) + (y*y) + (z*z));

	*x3 = x / dis;
	*y3 = y / dis;
	*z3 = z / dis;
}
