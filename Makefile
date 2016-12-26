IDIR=include
CFLAGS=-pedantic -std=c89 -Wall -Werror
ODIR=obj
LDIR=lib
SDIR=src
LIBS=-lm -lopencv_core -lopencv_highgui
CC=gcc
DEPS=

$(ODIR)/%.o: %.c %(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

spl-webcam-control: $(OBJ)
	gcc -o $@ $^ $(CFLAGS) $(LIBS) 