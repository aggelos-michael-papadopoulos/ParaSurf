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
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#ifndef FD_SET
#ifndef linux
#include <sys/select.h>
#endif
#endif
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>
#include <sys/uio.h>
#include "dms_param.h"
#include "atoms.h"
#include "wanted.h"
#include "protocol.h"

extern void	*emalloc();

#ifndef	FD_SET
#define	FD_SET(n, p)	((p)->fds_bits[0] |= (1 << (n)))
#define	FD_CLR(n, p)	((p)->fds_bits[0] &= ~(1 << (n)))
#define	FD_ISSET(n, p)	((p)->fds_bits[0] & (1 << n))
#define	FD_ZERO(p)	((p)->fds_bits[0] = 0)
#define	FD_ISZERO(p)	((p)->fds_bits[0] == 0)
#endif

#ifndef	DEBUG
#define STATIC static
#else
#define	STATIC
#endif

#ifndef	TRUE
#define	TRUE	1
#define	FALSE	0
#endif

typedef struct server_def	{
	struct server_def	*next;
	char			*host;
	FILE			*in_fd, *out_fd;
	int			state;
	void			(*reply)();
	void			(*retry)();
	int			arg1, arg2;
	int			nrequest;
}	SERVER;

#define	EXECUTING	0
#define	AVAILABLE	1
#define	DEAD		2
#define	IDLE		3
#define	STOPPED		4

STATIC NEIGHBOR	*neighbor;
STATIC PROBE	*probe;
STATIC SERVER	*server;
STATIC ATOM	**alist;
STATIC int	natom;

STATIC void	read_radii();
STATIC void	assign_radius();
STATIC void	print_astat();
STATIC int	compute_comp();
STATIC void	start_servers();
STATIC int	try_host();
STATIC void	stop_servers();
STATIC SERVER	*get_server();
STATIC void	wait_for_server();
STATIC void	print_sstat();
STATIC void	param_info();
STATIC void	pi_reply();
STATIC void	atom_info();
STATIC void	ai_reply();
STATIC void	ni_reply();
STATIC void	probe_info();
STATIC void	pri_reply();
STATIC void	neighbors();
STATIC void	c_reply();
STATIC void	probes();
STATIC void	p_reply();
STATIC void	contact_surface();
STATIC void	torus_surface();
STATIC void	probe_surface();
STATIC void	surface_reply();
STATIC void	collapse_probes();
STATIC void	cross();

extern char	(*resseqs)[RN_LEN];
extern int	numresseq;

/*
 * neighbor_info:
 *	Send neighbor information to all servers
 */
void neighbor_info(nlist, natom)
NEIGHBOR	*nlist;
int		natom;
{
	register SERVER		*sp;
	register NEIGHBOR	*np, *endnp;
	register struct iovec	*ip;
	register int		n;
	struct iovec		*countv, *datav;
	char			buf[BUFSIZ];

	endnp = nlist + natom;
	n = 0;
	countv = (struct iovec *) emalloc(natom * sizeof (struct iovec));
	for (np = nlist, ip = countv; np < endnp; np++, ip++) {
		ip->iov_base = (caddr_t) &np->nneighbor;
		ip->iov_len = sizeof (int);
		if (np->nneighbor > 0)
			n++;
	}
	if (n > 0) {
		datav = (struct iovec *) emalloc(n * sizeof (struct iovec));
		ip = datav;
		for (np = nlist; np < endnp; np++) {
			if (np->nneighbor <= 0)
				continue;
			ip->iov_base = (caddr_t) np->neighbor;
			ip->iov_len = np->nneighbor * sizeof (AINDEX);
			ip++;
		}
	}
	(void) sprintf(buf, NI_COMMAND, natom);
	for (sp = server; sp != NULL; sp = sp->next) {
		if (sp->state == DEAD)
			continue;
		fputs(buf, sp->out_fd);
		fwritev(sp->out_fd, countv, natom);
		if (n > 0)
			fwritev(sp->out_fd, datav, n);
		(void) fflush(sp->out_fd);
		sp->state = EXECUTING;
		sp->reply = ni_reply;
		sp->retry = NULL;
	}
	wait_for_server();
	(void) free((char *) countv);
	if (n > 0)
		(void) free((char *) datav);
}

/*
 * compute_ms:
 *	Compute the molecular surface
 */
void compute_ms(aarray, nat, radius, density, want_normal, verbose)
ATOM	**aarray;
int	nat;
double	radius, density;
int	want_normal;
int	verbose;
{
	register ATOM		**app, **endap;
	register PROBE		*pp;
	register int		i, j, k;

	/*
	 * Store the parameters so reply() routines can get at them
	 */
	alist = aarray;
	natom = nat;
	endap = alist + natom;

	/*
	 * Assign a radius for each atom
	 */
	read_radii();
	for (app = alist; app < endap; app++) {
		assign_radius(*app);
		(*app)->surface = NULL;
	}
	if (verbose)
		print_astat();

	/*
	 * Sort the atoms by coordinate so we can look just at slices
	 * instead of all atoms when looking for neighbors
	 */
	qsort((char *) alist, natom, sizeof (ATOM *), compute_comp);

	/*
	 * Start up servers on all the machines that we can reach
	 * and hand them the necessary parameters as well as atomic
	 * coordinates
	 */
	start_servers();
	param_info(radius, density, want_normal);
	atom_info(alist, natom);
	if (verbose)
		fputs("Servers initialized\n", stderr);

	/*
	 * Loop through the atom list and assign selected atoms to
	 * available servers (this should give us a list of neighbors).
	 * We then pass out the neighbor information to all the compute
	 * servers in preparation for surface computation.
	 */
	neighbor = (NEIGHBOR *) emalloc(natom * sizeof (NEIGHBOR));
	for (app = alist; app < endap; app++)
		neighbors(app - alist);
	wait_for_server();
	neighbor_info(neighbor, natom);
	if (verbose)
		fputs("Neighbors computed\n", stderr);

	/*
	 * Loop through the neighbor list and ask for probes which
	 * contact both neighbor simultaneously.  We then pass the
	 * probe information to all the compute servers as well.
	 */
	for (i = 0; i < natom; i++)
		for (j = 0; j < neighbor[i].nneighbor; j++)
			if (neighbor[i].neighbor[j] > i)
				probes(i, neighbor[i].neighbor[j]);
	wait_for_server();
	probe_info(probe);
	if (verbose)
		fputs("Probes computed\n", stderr);

	/*
	 * Now we clean up the neighbor info and send it out again
	 */
	for (i = 0; i < natom; i++) {
		for (j = 0, k = 0; j < neighbor[i].nneighbor; j++)
			if (neighbor[i].neighbor[j] >= 0)
				neighbor[i].neighbor[k++] =
					neighbor[i].neighbor[j];
		neighbor[i].nneighbor = k;
	}
#ifdef RETRANS_NB
	neighbor_info(neighbor, natom);
	if (verbose)
		fputs("Neighbors sent\n", stderr);
#endif

	/*
	 * Generate surface points
	 * First the contact, then the toroid reentrant,
	 * and finally spherical reentrant
	 */
	for (app = alist; app < endap; app++)
		if ((*app)->wanted)
			contact_surface(app - alist);
	for (i = 0; i < natom; i++)
		for (j = 0; j < neighbor[i].nneighbor; j++)
			if (neighbor[i].neighbor[j] > i) {
				if (!alist[i]->wanted
				&& !alist[neighbor[i].neighbor[j]]->wanted)
					continue;
				torus_surface(i, neighbor[i].neighbor[j]);
			}

	/*
	 * Collapse near-coincident probes and then compute surface area
	 */
	collapse_probes();
	for (pp = probe, i = 0; pp != NULL; pp = pp->next, i++) {
		if (!alist[pp->atom[0]]->wanted
		&&  !alist[pp->atom[1]]->wanted
		&&  !alist[pp->atom[2]]->wanted)
			continue;
		if (pp->real)
			probe_surface(i);
	}
	wait_for_server();
	if (verbose)
		fputs("Surface computed\n", stderr);

	/*
	 * Ah...we're done.  Terminate the servers and return
	 */
	if (verbose)
		print_sstat();
	stop_servers();
}

typedef struct radius_def	{
	struct radius_def	*next;
	char			*name;
	char			namelen;
	double			radius;
	int			natom;
}	RADIUS;

static RADIUS	*radius, *def_radius;

/*
 * read_radii:
 *	Get radii from radii file
 */
STATIC
void
read_radii()
{
	register RADIUS	*rp;
	register FILE	*fp;
	char		buf[BUFSIZ], name[10];
	double		length;

	if ((fp = fopen(RADII_FILE, "r")) == NULL) {
		if ((fp = fopen(DEF_RADII_FILE, "r")) == NULL) {
			perror(DEF_RADII_FILE);
			exit(1);
		}
	}
	while (fgets(buf, sizeof buf, fp) != NULL) {
		if (buf[0] == '#')
			continue;
		if (sscanf(buf, "%s%lf", name, &length) != 2) {
			fputs("Syntax error in radii file\n", stderr);
			fputs(buf, stderr);
			continue;
		}
		if (strcmp(name, "default") == 0) {
			def_radius = (RADIUS *) emalloc(sizeof (RADIUS));
			def_radius->next = NULL;
			def_radius->name = NULL;
			def_radius->namelen = 0;
			def_radius->radius = length;
			def_radius->natom = 0;
		}
		else {
			rp = (RADIUS *) emalloc(sizeof (RADIUS));
			rp->next = radius;
			rp->namelen = strlen(name);
			rp->name = emalloc(rp->namelen + 1);
			(void) strcpy(rp->name, name);
			rp->radius = length;
			rp->natom = 0;
			radius = rp;
		}
	}
	(void) fclose(fp);
}

/*
 * assign_radius:
 *	Assign a van der Waals radius to this atom
 */
STATIC
void
assign_radius(ap)
ATOM	*ap;
{
	register RADIUS	*rp;

	for (rp = radius; rp != NULL; rp = rp->next)
		if (strncmp(rp->name, ap->atname, rp->namelen) == 0)
			break;
	if (rp == NULL) {
		if (def_radius == NULL) {
			fprintf(stderr, "Unknown atom type %s\n", ap->atname);
			exit(1);
		}
		else
			rp = def_radius;
	}
	ap->radius = rp->radius;
	rp->natom++;
}

/*
 * print_astat:
 *	Print atom type statistics
 */
STATIC
void
print_astat()
{
	register struct radius_def	*rp;

	for (rp = radius; rp != NULL; rp = rp->next)
		if (rp->natom > 0)
			fprintf(stderr, "\t%s\t%d\n", rp->name, rp->natom);
	/*
	 * The last entry is the default
	 */
	if (def_radius != NULL && def_radius->natom > 0)
		fprintf(stderr, "\t???\t%d\n", def_radius->natom);
}

/*
 * compute_comp:
 *	Compare atoms based on coordinates
 */
STATIC
int
compute_comp(a1, a2)
ATOM	**a1, **a2;
{
	register ATOM	*t1, *t2;

	t1 = *a1;
	t2 = *a2;
	if (t1->coord[0] < t2->coord[0])
		return -1;
	else if (t1->coord[0] > t2->coord[0])
		return 1;
	if (t1->coord[1] < t2->coord[1])
		return -1;
	else if (t1->coord[1] > t2->coord[1])
		return 1;
	if (t1->coord[2] < t2->coord[2])
		return -1;
	else if (t1->coord[2] > t2->coord[2])
		return 1;
	return 0;
}

/*
 * start_servers:
 *	Start up servers on all available machines
 *	For the first pass implementation, we will try to contact
 *	known hosts only once and just use that set.  If hosts die,
 *	they get marked as down and are no longer used.  No attempts
 *	at retrying to connect to any host will be made.
 */
STATIC
void
start_servers()
{
	struct servent	*servp;

	if ((servp = getservbyname("dms", "tcp")) == NULL) {
		/* service entry not found, run in standalone mode */
		int     pid, toserver[2], fromserver[2];

		server = (SERVER *) emalloc(sizeof (SERVER));
		server->nrequest = 0;
		server->retry = NULL;
		server->host = "localhost";
		if (pipe(toserver) < 0 || pipe(fromserver) < 0) {
			perror("Cannot create pipe");
			exit(1);
		}
		if ((pid = fork()) < 0) {
			perror("Cannot fork");
			exit(1);
		}
		if (pid == 0) {
			close(0);
			close(1);
			dup2(toserver[0], 0);
			dup2(fromserver[1], 1);
			close(toserver[0]);
			close(toserver[1]);
			close(fromserver[0]);
			close(fromserver[1]);
			execl(SERVER_PATH, "dmsd", NULL);
			perror("execl");
			exit(1);
		}
		server->in_fd = fdopen(fromserver[0], "r");
		server->out_fd = fdopen(toserver[1], "w");
		close(fromserver[1]);
		close(toserver[0]);
		if (server->in_fd == NULL || server->out_fd == NULL) {
			fprintf(stderr, "Cannot open file descriptor.\n");
			exit(1);
		}
		setlinebuf(server->out_fd);
		server->state = AVAILABLE;
		server->next = NULL;
	} else {
		/* service entry found, run in distributed processing mode */
		register SERVER	*sp;
		register FILE	*fp;
		register char	*cp;
		char		buf[BUFSIZ];

		if ((fp = fopen(SERVER_FILE, "r")) == NULL) {
			if ((fp = fopen(DEF_SERVER_FILE, "r")) == NULL) {
				perror(DEF_SERVER_FILE);
				exit(1);
			}
		}
		sethostent(TRUE);
		while (fgets(buf, sizeof buf, fp) != NULL) {
			if ((cp = strchr(buf, '\n')) != NULL)
				*cp = '\0';
			sp = (SERVER *) emalloc(sizeof (SERVER));
			sp->host = emalloc(strlen(buf) + 1);
			sp->nrequest = 0;
			sp->retry = NULL;
			(void) strcpy(sp->host, buf);
			if (try_host(buf, sp, servp->s_port) < 0)
				(void) free((char *) sp);
			else {
				sp->next = server;
				server = sp;
			}
		}
		endhostent();
		(void) fclose(fp);
	}
}

/*
 * try_host:
 *	Try to contact a host.
 */
STATIC
int
try_host(host, sp, port)
char	*host;
SERVER	*sp;
int	port;
{
	struct hostent		*hostp;
	struct sockaddr_in	addr;
	int			fd;
	int			newfd;
	extern int		errno;

	if ((hostp = gethostbyname(host)) == NULL) {
		fprintf(stderr, "%s: no such host\n", host);
		return -1;
	}
	if ((fd = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
		perror("socket");
		return -1;
	}
	addr.sin_family = PF_INET;
	addr.sin_addr = *((struct in_addr *) hostp->h_addr);
	addr.sin_port = port;
	if (connect(fd, (struct sockaddr *) &addr, sizeof addr) < 0) {
		perror(host);
		if (errno != ETIMEDOUT) {
			(void) close(fd);
			return -1;
		}
		else {
			(void) close(fd);
			sp->state = DEAD;
			return 0;
		}
	}

	if ((sp->in_fd = fdopen(fd, "r")) == NULL) {
		perror("fdopen");
		(void) close(fd);
		sp->state = DEAD;
		return 0;
	}

#ifdef BSD
	newfd = fd;
#else
	newfd = dup(fd);
#endif
	if ((sp->out_fd = fdopen(newfd, "w")) == NULL) {
		perror("fdopen");
		(void) close(fd);
		sp->state = DEAD;
		return 0;
	}

	sp->state = AVAILABLE;
	return 0;
}

/*
 * stop_servers:
 *	Terminate all servers that are currently active
 */
STATIC
void
stop_servers()
{
	register SERVER	*sp, *nextsp;

	for (sp = server; sp != NULL; sp = nextsp) {
		nextsp = sp->next;
		if (sp->state != DEAD) {
			(void) fclose(sp->in_fd);
			(void) fclose(sp->out_fd);
		}
		(void) free((char *) sp);
	}
}

/*
 * get_server:
 *	Get an available server.  Bomb out if all servers are dead.
 */
STATIC
SERVER *
get_server()
{
	register SERVER	*sp;
	register int	max_fd, any_live, n;
	fd_set		mask, save_mask;

	max_fd = 0;	/* In case _file is declared unsigned */
	any_live = FALSE;
	FD_ZERO(&save_mask);
	n = 0;

	/*
	 * If there is an available server, return it.  Otherwise,
	 * keep track of the list of currently executing servers
	 */
	for (sp = server; sp != NULL; sp = sp->next) {
		switch (sp->state) {
		  case AVAILABLE:
			sp->nrequest++;
			return sp;
		  case EXECUTING:
		  case STOPPED:
			FD_SET(fileno(sp->in_fd), &save_mask);
			n++;
			if (fileno(sp->in_fd) > max_fd)
				max_fd = fileno(sp->in_fd);
			any_live = TRUE;
			break;
		  case IDLE:
			any_live = TRUE;
			break;
		  case DEAD:
			break;
		}
	}

	if (!any_live) {
		fputs("All servers died!\n", stderr);
		exit(1);
	}
	if (n <= 0)
		return NULL;
	max_fd++;

again:
	mask = save_mask;
	/*
	 * Wait for a server to come back (either respond or die)
	 */
	if (select(max_fd, &mask, (fd_set *) NULL, (fd_set *) NULL,
	(struct timeval *) NULL) <= 0) {
		perror("select");
		exit(1);
	}
	for (sp = server; sp != NULL; sp = sp->next) {
		if (sp->state != EXECUTING && sp->state != STOPPED)
			continue;
		if (!FD_ISSET(fileno(sp->in_fd), &mask))
			continue;
		(*sp->reply)(sp);
		switch (sp->state) {
		  case AVAILABLE:
			sp->nrequest++;
			return sp;
		  case EXECUTING:
		  case STOPPED:
			break;
		  default:
			FD_CLR(fileno(sp->in_fd), &save_mask);
			n--;
			break;
		}
	}
	if (n <= 0) {
		fputs("All servers died!\n", stderr);
		exit(1);
	}
	goto again;
}

/*
 * wait_for_server:
 *	Wait for all servers to become idle.
 *	If there are servers that died while handling requests that
 *	need to be retried, handle that as well.  Since servers normally
 *	do not die, we are willing to take a performance penalty to handle
 *	the special case.
 */
STATIC
void
wait_for_server()
{
	register SERVER	*sp;
	register int	n;

again:
	/*
	 * Wait for all active servers to finish
	 */
	while ((sp = get_server()) != NULL) {
		sp->state = IDLE;
		continue;
	}
	for (sp = server; sp != NULL; sp = sp->next)
		if (sp->state == IDLE)
			sp->state = AVAILABLE;

	/*
	 * Check for servers that died in action.  If they need to
	 * be retried, start them up and go back to the waiting stage
	 */
	n = 0;
	for (sp = server; sp != NULL; sp = sp->next)
		if (sp->state == DEAD && sp->retry != NULL) {
			(*sp->retry)(sp->arg1, sp->arg2);
			sp->retry = NULL;
#ifdef DEBUG
			fprintf(stderr, "Retrying after %s died\n", sp->host);
#endif
			n++;
		}
	if (n > 0)
		goto again;
}

/*
 * print_sstat:
 *	Print server statistics
 */
STATIC
void
print_sstat()
{
	register SERVER	*sp;

	fputs("Server request count:\n", stderr);
	for (sp = server; sp != NULL; sp = sp->next)
		fprintf(stderr, "\t%-8s\t%d\n", sp->host, sp->nrequest);
}

/*
 * wanted:
 *	Determine whether the given atom is in the "wanted list"
 *	We "know" we are called with non-null arguments
 */
int
wanted(ATOM *ap, WANTED *wlist)
{
	register WANTED	*wp;
	int		from, to, at;

	at = ap->resindex;
	for (wp = wlist; wp != NULL; wp = wp->next)
		switch (wp->type) {
		  case W_RANGE:
			from = wp->startres;
			to = wp->w.endres;
			if (from < 0 || to < 0)
				continue;
			if (from <= at && at <= to)
				return TRUE;
			break;
		  case W_ANY:
			if (at == wp->startres)
				return TRUE;
			break;
		  case W_ATOM:
			if (at != wp->startres)
				continue;
			if (strcmp(ap->atname, wp->w.atom) == 0)
				return TRUE;
			break;
		}
	return FALSE;
}

/*
 * locres:
 *	return the numeric index of the given resseq in the array of
 *	all resseqs (which should be in the same order as the PDB file).
 *	Returns -1 if not found.
 */
int
locres(seq)
char	*seq;
{
	register int	i;
	int		index;

	index = -1;
	for (i = 0; i < numresseq; i++) {
		if (strcmp(resseqs[i], seq) == 0) {
			index = i;
			break;
		}
	}
	return index;
}

/*
 * param_info:
 *	Pass parameter (probe radius and whether we need normals)
 *	to servers
 */
STATIC
void
param_info(radius, density, want_normal)
double	radius, density;
int	want_normal;
{
	register SERVER	*sp;
	char		buf[BUFSIZ];

	(void) sprintf(buf, PI_COMMAND, radius, density, want_normal);
	for (sp = server; sp != NULL; sp = sp->next) {
		if (sp->state == DEAD)
			continue;
		fputs(buf, sp->out_fd);
		(void) fflush(sp->out_fd);
		sp->state = EXECUTING;
		sp->reply = pi_reply;
		sp->retry = NULL;
	}
	wait_for_server();
}

/*
 * pi_reply:
 *	Wait for response to param info command
 */
STATIC
void
pi_reply(sp)
SERVER	*sp;
{
	char		buf[BUFSIZ];

	if (fgets(buf, sizeof buf, sp->in_fd) == NULL) {
		fputs("server died\n", stderr);
		sp->state = DEAD;
		(void) fclose(sp->in_fd);
		(void) fclose(sp->out_fd);
		return;
	}
	if (strcmp(buf, PI_RESPONSE) != 0) {
		fputs("pi_reply: unexpected response from server\n", stderr);
		fputs(buf, stderr);
		sp->state = DEAD;
		(void) fclose(sp->in_fd);
		(void) fclose(sp->out_fd);
		return;
	}
	sp->state = AVAILABLE;
}

/*
 * atom_info:
 *	Pass atom information out to all servers
 */
STATIC
void
atom_info(alist, natom)
ATOM	**alist;
int	natom;
{
	register SERVER	*sp;
	register ATOM	**app;
	struct iovec	*iov, *ip, *endip;
	char		buf[BUFSIZ];

	iov = (struct iovec *) emalloc(natom * sizeof (struct iovec));
	endip = iov + natom;
	app = alist;
	for (ip = iov; ip < endip; ip++) {
		ip->iov_base = (caddr_t) *app++;
		ip->iov_len = sizeof (ATOM);
	}
	(void) sprintf(buf, AI_COMMAND, natom);
	for (sp = server; sp != NULL; sp = sp->next) {
		if (sp->state == DEAD)
			continue;
		(void) fputs(buf, sp->out_fd);
		fwritev(sp->out_fd, iov, natom);
		(void) fflush(sp->out_fd);
		sp->state = EXECUTING;
		sp->reply = ai_reply;
		sp->retry = NULL;
	}
	wait_for_server();
	(void) free((char *) iov);
}

/*
 * ai_reply:
 *	Read a reply from a server that just received atom info
 */
STATIC
void
ai_reply(sp)
SERVER	*sp;
{
	char		buf[BUFSIZ];

	if (fgets(buf, sizeof buf, sp->in_fd) == NULL) {
		fputs("ai_reply: server died\n", stderr);
		goto bad;
	}
	if (strcmp(buf, AI_RESPONSE) != 0) {
		fputs("ai_reply: unexpected response from server\n", stderr);
		fputs(buf, stderr);
		goto bad;
	}
	sp->state = AVAILABLE;
	return;
bad:
	sp->state = DEAD;
	(void) fclose(sp->in_fd);
	(void) fclose(sp->out_fd);
	return;
}

/*
 * ni_reply:
 *	Read a reply from a server that just received neighbor info
 */
STATIC
void
ni_reply(sp)
SERVER	*sp;
{
	char		buf[BUFSIZ];

	if (fgets(buf, sizeof buf, sp->in_fd) == NULL) {
		fputs("ni_reply: server died\n", stderr);
		goto bad;
	}
	if (strcmp(buf, NI_RESPONSE) != 0) {
		fputs("ni_reply: unexpected response from server\n", stderr);
		fputs(buf, stderr);
		goto bad;
	}
	sp->state = AVAILABLE;
	return;
bad:
	sp->state = DEAD;
	(void) fclose(sp->in_fd);
	(void) fclose(sp->out_fd);
	return;
}


/*
 * probe_info:
 *	Send probe information to all servers
 */
STATIC
void
probe_info(plist)
PROBE	*plist;
{
	register SERVER		*sp;
	register PROBE		*pp;
	register int		n;
	register struct iovec	*ip;
	struct iovec		*iov, *endip;
	char			buf[BUFSIZ];

	/*
	 * Count up the number of probes and allocate
	 * data structure for sending it down the line
	 */
	n = 0;
	for (pp = probe; pp != NULL; pp = pp->next)
		n++;
	if (n > 0) {
		iov = (struct iovec *) emalloc(n * sizeof (struct iovec));
		endip = iov + n;
		for (ip = iov, pp = plist; ip < endip; ip++, pp = pp->next) {
			ip->iov_base = (caddr_t) pp;
			ip->iov_len = sizeof (PROBE);
		}
	}

	(void) sprintf(buf, PRI_COMMAND, n);
	for (sp = server; sp != NULL; sp = sp->next) {
		if (sp->state == DEAD)
			continue;
		fputs(buf, sp->out_fd);
		if (n > 0)
			fwritev(sp->out_fd, iov, n);
		(void) fflush(sp->out_fd);
		sp->state = EXECUTING;
		sp->reply = pri_reply;
		sp->retry = NULL;
	}
	wait_for_server();
	if (n > 0)
		(void) free((char *) iov);
}

/*
 * pri_reply:
 *	Read a reply from a server that just received probe info
 */
STATIC
void
pri_reply(sp)
SERVER	*sp;
{
	char		buf[BUFSIZ];

	if (fgets(buf, sizeof buf, sp->in_fd) == NULL) {
		fputs("pri_reply: server died\n", stderr);
		goto bad;
	}
	if (strcmp(buf, PRI_RESPONSE) != 0) {
		fputs("pri_reply: unexpected response from server\n", stderr);
		fputs(buf, stderr);
		goto bad;
	}
	sp->state = AVAILABLE;
	return;
bad:
	sp->state = DEAD;
	(void) fclose(sp->in_fd);
	(void) fclose(sp->out_fd);
	return;
}

/*
 * neighbors:
 *	Send contact computation request for atom "n" in "alist"
 *	out to an available server.  If none is available, wait for one.
 */
STATIC
void
neighbors(n)
int	n;
{
	register SERVER	*sp;

	if ((sp = get_server()) == NULL) {
		fputs("neighbors: servers idling!\n", stderr);
		exit(1);
	}
	fprintf(sp->out_fd, C_COMMAND, n);
	(void) fflush(sp->out_fd);
	sp->state = EXECUTING;
	sp->reply = c_reply;
	sp->retry = neighbors;
	sp->arg1 = n;
}

/*
 * c_reply:
 *	Read the output of a server which completed a contact
 *	computation.  Reset the server state when we are done.
 */
STATIC
void
c_reply(sp)
SERVER	*sp;
{
	char		buf[BUFSIZ];
	int		n, nn;

	/*
	 * Read the computed neighbors
	 */
	if (fgets(buf, sizeof buf, sp->in_fd) == NULL) {
		fputs("c_reply: server died\n", stderr);
		goto bad;
	}
	if (sscanf(buf, C_RESPONSE, &n, &nn) != 2) {
		fputs("c_reply: unexpected response from server\n", stderr);
		fputs(buf, stderr);
		goto bad;
	}
	neighbor[n].nneighbor = nn;
	if (nn > 0) {
		neighbor[n].neighbor = (AINDEX *) emalloc(nn * sizeof (AINDEX));
		if (fread((char *) neighbor[n].neighbor, sizeof (AINDEX), nn,
		sp->in_fd) != nn) {
			fputs("c_reply: unexpected EOF from server\n", stderr);
			goto bad;
		}
	}
	else
		neighbor[n].neighbor = NULL;

	/*
	 * Reset the server state and return
	 */
	sp->state = AVAILABLE;
	return;

bad:
	/*
	 * Mark the server as dead and return
	 */
	sp->state = DEAD;
	(void) fclose(sp->in_fd);
	(void) fclose(sp->out_fd);
	return;
}

/*
 * probes:
 *	Send contact computation request for atom "n" in "alist"
 *	out to an available server.  If none is available, wait for one.
 */
STATIC
void
probes(n, m)
int	n;
AINDEX	m;
{
	register SERVER	*sp;

	if ((sp = get_server()) == NULL) {
		fputs("neighbors: servers idling!\n", stderr);
		exit(1);
	}
	fprintf(sp->out_fd, P_COMMAND, n, m);
	(void) fflush(sp->out_fd);
	sp->state = EXECUTING;
	sp->reply = p_reply;
	sp->retry = probes;
	sp->arg1 = n;
	sp->arg2 = m;
}

/*
 * p_reply:
 *	Read the output of a server which completed a probe
 *	computation.  Reset the server state when we are done.
 */
STATIC
void
p_reply(sp)
SERVER	*sp;
{
	register int	i;
	register PROBE	*pp;
	char		buf[BUFSIZ];
	int		n, m, np;

	for (;;) {
		if (fgets(buf, sizeof buf, sp->in_fd) == NULL) {
			fputs("p_reply: server died\n", stderr);
			goto bad;
		}
		if (sscanf(buf, P_BADNB, &n, &m) != 2)
			break;
		/*
		 * These atoms are not real neighbors!  We should go and
		 * remove the neighbor markings on them.
		 */
		for (i = 0; i < neighbor[n].nneighbor; i++)
			if (neighbor[n].neighbor[i] == m)
				neighbor[n].neighbor[i] = -1;
		for (i = 0; i < neighbor[m].nneighbor; i++)
			if (neighbor[m].neighbor[i] == n)
				neighbor[m].neighbor[i] = -1;
	}
	if (sscanf(buf, P_RESPONSE, &np) != 1) {
		fputs("p_reply: unexpected response from server\n",
			stderr);
		fputs(buf, stderr);
		goto bad;
	}
	if (np > 0) {
		pp = (PROBE *) emalloc(np * sizeof (PROBE));
		if (fread((char *) pp, sizeof (PROBE), np, sp->in_fd) != np) {
			fputs("p_reply: unexpected EOF from server\n", stderr);
			goto bad;
		}
		for (i = 1; i < np; i++)
			pp[i - 1].next = &pp[i];
		pp[np - 1].next = probe;
		probe = pp;
	}

	/*
	 * Reset the server state and return
	 */
	sp->state = AVAILABLE;
	return;

bad:
	/*
	 * Mark the server as dead and return
	 */
	sp->state = DEAD;
	(void) fclose(sp->in_fd);
	(void) fclose(sp->out_fd);
	return;
}

/*
 * contact_surface:
 *	Start a server on computing the contact surface for atom `n'
 */
STATIC
void
contact_surface(n)
int	n;
{
	register SERVER	*sp;

	if ((sp = get_server()) == NULL) {
		fputs("contact_surface: servers idling!\n", stderr);
		exit(1);
	}
	fprintf(sp->out_fd, CS_COMMAND, n);
	(void) fflush(sp->out_fd);
	sp->state = EXECUTING;
	sp->reply = surface_reply;
	sp->retry = contact_surface;
	sp->arg1 = n;
}

/*
 * torus_surface:
 *	Start a server on computing the toroid reentrant surface between
 *	atoms `n' and `m'
 */
STATIC
void
torus_surface(n, m)
int	n, m;
{
	register SERVER	*sp;

	if ((sp = get_server()) == NULL) {
		fputs("torus_surface: servers idling!\n", stderr);
		exit(1);
	}
	fprintf(sp->out_fd, TS_COMMAND, n, m);
	(void) fflush(sp->out_fd);
	sp->state = EXECUTING;
	sp->reply = surface_reply;
	sp->retry = torus_surface;
	sp->arg1 = n;
	sp->arg2 = m;
}

/*
 * probe_surface:
 *	Start a server on computing the probe reentrant surface
 *	for probe `pp'
 */
STATIC
void
probe_surface(n)
int	n;
{
	register SERVER	*sp;

	if ((sp = get_server()) == NULL) {
		fputs("probe_surface: servers idling!\n", stderr);
		exit(1);
	}
	fprintf(sp->out_fd, PS_COMMAND, n);
	(void) fflush(sp->out_fd);
	sp->state = EXECUTING;
	sp->reply = surface_reply;
	sp->retry = probe_surface;
	sp->arg1 = n;
}

/*
 * surface_reply:
 *	This routine is used by all surface computation requests.
 *	We read surface data until we get an "END" message
 */
STATIC
void
surface_reply(sp)
SERVER	*sp;
{
	register SURFACE	*surfp;
	register int		cc;
	char			buf[BUFSIZ];
	int			n, nn, type, have_normal;
	double			area;

	for (;;) {
		if (fgets(buf, sizeof buf, sp->in_fd) == NULL) {
			fputs("surface_reply: server died\n", stderr);
			goto bad;
		}
		if (strcmp(buf, SURFACE_END) == 0)
			break;
		if (sscanf(buf, SURFACE_RESPONSE, &n, &nn, &type, &area,
		&have_normal) != 5) {
			fputs("surface_reply: bad response from server\n",
				stderr);
			fputs(buf, stderr);
			goto bad;
		}
		if (nn <= 0)
			continue;
		surfp = (SURFACE *) emalloc(sizeof (SURFACE));
		surfp->next = alist[n]->surface;
		alist[n]->surface = surfp;
		surfp->type = type;
		surfp->area = area;
		surfp->npoint = nn;
		surfp->position = (POINT *) emalloc(nn * sizeof (POINT));
		if ((cc = fread((char *) surfp->position, sizeof (POINT), nn,
		sp->in_fd)) != nn) {
			fprintf(stderr,
				"surface_reply: Read %d points, wanted %d\n",
				cc, nn);
			surfp->npoint = 0;
			goto bad;
		}
		if (!have_normal) {
			surfp->normal = NULL;
			continue;
		}
		surfp->normal = (POINT *) emalloc(nn * sizeof (POINT));
		if (fread((char *) surfp->normal, sizeof (POINT), nn,
		sp->in_fd) != nn) {
			fputs("surface_reply: EOF from server\n", stderr);
			surfp->npoint = 0;
			goto bad;
		}
	}

	/*
	 * Got all data, just reset state and return
	 */
	sp->state = AVAILABLE;
	return;

bad:
	/*
	 * Mark the server as dead and return
	 */
	sp->state = DEAD;
	(void) fclose(sp->in_fd);
	(void) fclose(sp->out_fd);
	return;
}

/*
 * collapse_probes:
 *	Check for near-coincident probes and remove redundant
 *	surface coverage
 */
STATIC
void
collapse_probes()
{
	int	i, j;
	PROBE	*pp, *npp;
	double	delta, distsq;
	ATOM	*shared[3];
	int	nshared;
	double	plane[4];
	double	v1[3], v2[3];
	double	pside, npside;
#define	EPSILON	1e-8

	for (pp = probe; pp != NULL; pp = pp->next) {
		if (!pp->real)
			continue;
		for (npp = pp->next; npp != NULL; npp = npp->next) {
			distsq = 0;
			for (i = 0; i < 3; i++) {
				delta = pp->coord[i] - npp->coord[i];
				distsq += delta * delta;
			}
			if (distsq > EPSILON)
				continue;
			nshared = 0;
			for (i = 0; i < 3; i++)
				for (j = 0; j < 3; j++)
					if (pp->atom[i] == npp->atom[j]) {
						shared[nshared++] =
							alist[pp->atom[i]];
						break;
					}
			if (nshared != 2)
				continue;
			for (i = 0; i < 3; i++) {
				v1[i] = shared[0]->coord[i] - pp->coord[i];
				v2[i] = shared[1]->coord[i] - pp->coord[i];
			}
			cross(v1, v2, plane);
			plane[3] = 0;
			for (i = 0; i < 3; i++)
				plane[3] -= plane[i] * pp->coord[i];
			for (i = 0; i < 3; i++) {
				if (alist[pp->atom[i]] != shared[0]
				&& alist[pp->atom[i]] != shared[1]) {
					pside = plane[3];
					for (j = 0; j < 3; j++)
						pside += alist[pp->atom[i]]
							->coord[j] * plane[j];
				}
				if (alist[npp->atom[i]] != shared[0]
				&& alist[npp->atom[i]] != shared[1]) {
					npside = plane[3];
					for (j = 0; j < 3; j++)
						npside += alist[npp->atom[i]]
							->coord[j] * plane[j];
				}
			}
			if ((npside < 0 && pside < 0)
			|| (npside > 0 && pside > 0))
				npp->real = 0;
		}
	}
}

/*
 * cross:
 *	Take cross product of a and b
 */
STATIC
void
cross(a, b, result)
double a[3];
double b[3];
double result[3];
{
	result[0] = a[1] * b[2] - b[1] * a[2];
	result[1] = a[2] * b[0] - b[2] * a[0];
	result[2] = a[0] * b[1] - b[0] * a[1];
}
