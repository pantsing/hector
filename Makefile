PREFIX=/usr/local
DESTDIR=
GOFLAGS=
BINDIR=${PREFIX}/bin

BLDDIR = build

APPS = hector
all: $(APPS)

$(BLDDIR)/hector:   $(wildcard hector.go internal/cmds/run/*.go internal/global.go internal/version/*.go internal/algorithms/*/*.go internal/core/*.go internal/utils/*.go)

$(BLDDIR)/%:
	@mkdir -p $(dir $@)
	go build ${GOFLAGS} -o $@ ../$*

$(APPS): %: $(BLDDIR)/%

clean:
	rm -fr $(BLDDIR)

.PHONY: install clean all
.PHONY: $(APPS)

install: $(APPS)
	install -m 755 -d ${DESTDIR}${BINDIR}
	for APP in $^ ; do install -m 755 ${BLDDIR}/$$APP ${DESTDIR}${BINDIR}/$$APP ; done