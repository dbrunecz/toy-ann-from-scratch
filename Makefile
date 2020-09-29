CFLAGS+=-Wall -Werror
CFLAGS+=-ggdb

LDLIBS+=-lm

ann:

clean:
	@rm -f ann 2>/dev/null
