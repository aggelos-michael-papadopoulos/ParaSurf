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

/*
 * This function is like the one from the E & S library which produces
 * a matrix to rotate a vector to the Z axis.
 *
 * This version is much enhanced over the Pic Sys original, in that
 * you give it both ends of the bond to be rotated about and it
 * returns both the "m_lookat" tensor and its inverse.
 *
 * Martin Pensak:  1977
 *
 * Stolen and hacked for the PS300 by Conrad Huang 24feb84
 */
lookat(array, arrayinv, x, y)
double	array[4][4];		/* the output array */
double	arrayinv[4][4];		/* output array for the inverse of
				 * the main output */
double	x[3], y[3];		/* the vectors */
{
	register int	j, k;
	double		a, b, c, l, d;
	double		m[4][4];

	a = y[0] - x[0];
	b = y[1] - x[1];
	c = y[2] - x[2];
	l = sqrt(a*a + c*c);
	d = sqrt(l*l + b*b);
	if (d == 0.0) {
		fprintf(stderr, "Illegal value passed to m_lookat\n");
		return;
	}

	m[0][0] = (l != 0) ? (c / l) : 1;
	m[0][1] = (l != 0) ? (-(a * b)/(l * d)) : 0;
	m[0][2] = a / d;
	m[0][3] = 0;
	m[1][0] = 0;
	m[1][1] = l / d;
	m[1][2] = b / d;
	m[1][3] = 0;
	m[2][0] = (l != 0) ? ( - a / l) : 0;
	m[2][1] = (l != 0) ? ( -(b * c)/(l * d)) : 1;
	m[2][2] = c / d;
	m[2][3] = 0;
	m[3][0] = m[3][1] = m[3][2] = 0;
	m[3][3] = 1;

	for (j = 0; j < 4; j++)
		for (k = 0; k < 4; k++)
			array[j][k] = arrayinv[k][j] = m[j][k];

	/* now set up the translations */
	a = x[0];
	b = x[1];
	c = x[2];
	arrayinv[3][0] = a;
	arrayinv[3][1] = b;
	arrayinv[3][2] = c;

	array[3][0] = -(a*m[0][0] + b*m[1][0] + c*m[2][0]);
	array[3][1] = -(a*m[0][1] + b*m[1][1] + c*m[2][1]);
	array[3][2] = -(a*m[0][2] + b*m[1][2] + c*m[2][2]);
}
