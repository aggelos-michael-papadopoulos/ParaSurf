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
#include <stdlib.h>

/*
 * emalloc:
 *	Same as malloc but exits when failure occurs
 */
char *
emalloc(size)
unsigned	size;
{
	register char	*cp;

	if ((cp = malloc(size)) == NULL) {
		fprintf(stderr, "emalloc(%u) failed\n", size);
		exit(1);
	}
	return cp;
}

/*
 * erealloc:
 *	Same as realloc but exits when failure occurs
 */
char *
erealloc(ptr, size)
char		*ptr;
unsigned	size;
{
	register char	*cp;

	if ((cp = realloc(ptr, size)) == NULL) {
		fprintf(stderr, "erealloc(ptr, %u) failed\n", size);
		exit(1);
	}
	return cp;
}

/*
 * ecalloc:
 *	Same as calloc but exits when failure occurs
 */
char *
ecalloc(nelem, elsize)
unsigned nelem, elsize;
{
	register char	*cp;

	if ((cp = calloc(nelem, elsize)) == NULL) {
		fprintf(stderr, "calloc(%u, %u) failed\n", nelem, elsize);
		exit(1);
	}
	return cp;
}
