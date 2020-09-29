CFLAGS+=-Wall -Werror
CFLAGS+=-ggdb

LDLIBS+=-lm

test: ann
	./ann

ann:

clean:
	@rm -f ann 2>/dev/null
