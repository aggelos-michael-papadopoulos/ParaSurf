/*
 *	Copyright (c) 1989 The Regents of the University of California.
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
 *	$Id: pdb_sscanf.c,v 2.8 1994/04/15 22:34:24 gregc Exp $
 */

/* LINTLIBRARY */

# include	<stdio.h>
# include	<ctype.h>
# ifdef __STDC__
# include	<stdarg.h>
# include	<stdlib.h>
# else
# include	<varargs.h>

extern int	atoi();
extern double	atof();
# endif

/*
 *	pdb_sscanf performs similarly to sscanf, execept that fields are of
 *	fixed length and a complete line is always consumed.  The field
 *	width defaults to one.  If the line is shorter than expected then
 *	the default is returned.
 *
 *		d	get an integer.  Default:  0.
 *		f	get a floating point number (C double).  Default:  0.0.
 *		(space) ignore characters within field
 *		s	get a C string, leading and trailing spaces are
 *			stripped; the field width is used as a limit on
 *			the string length, the null character is appended
 *			to the end of the string.  Default:  empty string.
 *		c	get a character(s); no stripping of spaces, nor is
 *			a null character appended.  Default:  space(s).
 */

# define	MAXFIELDSIZE	64

int
# ifdef __STDC__
pdb_sscanf(const char *buffer, const char *fmt, ...)
# else
/*VARARGS2*/
pdb_sscanf(buffer, fmt, va_alist)
	char	*buffer;
	char	*fmt;
	va_dcl
# endif
{
	va_list	ap;
	int	i, field_width;
	int	nmatch;
	char	*s, *t;
	char	tmp[MAXFIELDSIZE];

# ifdef __STDC__
	va_start(ap, fmt);
# else
	va_start(ap);
# endif
	nmatch = 0;
	for (; *fmt != '\0'; fmt++) {
		if (*fmt != '%') {
			if (*buffer == *fmt)
				buffer++;
			else if (*buffer != '\0' && *buffer != '\n')
				return -1;
			continue;
		}

		/* calculate field_width */
		field_width = 0;
		for (++fmt; isdigit(*fmt); fmt++)
			field_width = field_width * 10 + *fmt - '0';
		if (field_width == 0)
			field_width = 1;	/* default */
		if (*buffer != '\0' && *buffer != '\n')
			nmatch++;

		switch (*fmt) {

		case 'd':			/* integer */
			/* if we've already seen the end of the buffer, don't
			   try to get anymore characters */
			if (*buffer == '\0' || *buffer == '\n') {
				*(va_arg(ap, int *)) = 0;
				break;
			}

			s = tmp;
			for (i = 0; i < field_width; i++) {
				if (*buffer == '\0' || *buffer == '\n')
					break;
				*s++ = *buffer++;
			}
			*s = '\0';
#ifndef __STDC__
			*(va_arg(ap, int *)) = atoi(tmp);
#else
			/* remove trailing spaces */
			while (s > tmp && isspace(*(s - 1)))
				*--s = '\0';
			*(va_arg(ap, int *)) = (int) strtol(tmp, &t, 10);
			if (t != s)
				return -1;
#endif
			break;

		case 'f':			/* floating point */
			/* if we've already seen the end of the buffer, don't
			   try to get anymore characters */
			if (*buffer == '\0' || *buffer == '\n') {
				*(va_arg(ap, double *)) = 0.0;
				break;
			}

			s = tmp;
			for (i = 0; i < field_width; i++) {
				if (*buffer == '\0' || *buffer == '\n')
					break;
				*s++ = *buffer++;
			}
			*s = '\0';
#ifndef __STDC__
			*(va_arg(ap, double *)) = atof(tmp);
#else
			/* remove trailing spaces */
			while (s > tmp && isspace(*(s - 1)))
				*--s = '\0';
			*(va_arg(ap, double *)) = strtod(tmp, &t);
			if (t != s)
				return -1;
#endif
			break;

		case 's':			/* string */
			/* if we've already seen the end of the buffer, don't
			   try to get anymore characters */
			if (*buffer == '\0' || *buffer == '\n') {
				*(va_arg(ap, char *)) = '\0';
				break;
			}

			s = t = va_arg(ap, char *);
			for (i = 0; i < field_width; i++) {
				if (*buffer == '\0' || *buffer == '\n')
					break;
				*s++ = *buffer++;
			}
			*s = '\0';
			/* remove trailing spaces */
			while (s > t && isspace(*--s))
				*s = '\0';
			break;

		case 'c':			/* character(s) */
			s = va_arg(ap, char *);
			for (i = 0; i < field_width; i++)
				s[i] = ' ';	/* default */

			/* if we've already seen the end of the buffer, don't
			   try to get anymore characters */
			if (*buffer == '\0' || *buffer == '\n')
				break;

			for (i = 0; i < field_width; i++) {
				if (*buffer == '\0' || *buffer == '\n')
					break;
				*s++ = *buffer++;
			}
			break;

		case ' ':			/* space (ignore) */
			/* if we've already seen the end of the buffer, don't
			   try to get anymore characters */
			if (*buffer == '\0' || *buffer == '\n')
				break;

			for (i = 0; i < field_width; i++, buffer++)
				if (*buffer == '\0' || *buffer == '\n')
					break;
			break;

		default:
			fprintf(stderr, "bad format '%c' in pdb_sscanf\n",
									*fmt);
			va_end(ap);
			return -1;
		}
	}
	va_end(ap);
	return nmatch;
}
