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
#include	<stdio.h>
#include	<ctype.h>

/*
 * tokenize:
 *	Break string into array of words at spaces, keeping strings
 *	intact and recognizing backslash escapes, returning number
 *	of words found or -1 if more than n found or -2 if non-printable
 *	character encountered or -3 if mismatched quotes.
 */
int
tokenize(string, array, n)
char	*string,	/* string to be tokenized */
	*array[];	/* returned array of pointers to tokens in string */
int	n;		/* maximum number of tokens to look for */
{
	register int	i;

	for (i = 0; i < n; i++) {
		while (isspace(*string))
			string++;
		if (*string == '"') {
			*array++ = ++string;
			while (isprint(*string) && *string != '"') {
				if (*string == '\\') { /* backslash escapes */
					strcpy(string, string+1);
					switch (*string) {
					  case 't': /* tab */
						*string = '\t';
						break;
					  case 'b': /* backspace */
						*string = '\b';
						break;
					  case 'n': /* newline */
						*string = '\n';
						break;
					  case 'r': /* carriage return */
						*string = '\r';
						break;
					  case 'f': /* formfeed */
						*string = '\f';
						break;
					  default: /* treat as normal */
						break;
					}
				}
				string++;
			}
			if (*string == '\0')
				return -3;
			if (!isprint(*string))
				return -2;
			*string++ = '\0';
			continue;
		}
		*array++ = string;
		if (*string == '\0')
			break;
		if (!isprint(*string))
			return -2;
		while (!isspace(*string) && isprint(*string)) {
			if (*string == '\\') { /* backslash escapes */
				strcpy(string, string+1);
				switch (*string) {
				  case 't': /* tab */
					*string = '\t';
					break;
				  case 'b': /* backspace */
					*string = '\b';
					break;
				  case 'n': /* newline */
					*string = '\n';
					break;
				  case 'r': /* carriage return */
					*string = '\r';
					break;
				  case 'f': /* formfeed */
					*string = '\f';
					break;
				  default: /* treat as normal */
					break;
				}
			}
			string++;
		}
		if (isspace(*string))
			*string++ = '\0';
	}
	while (isspace(*string))
		string++;
	return *string == '\0' ? i : -1;
}
