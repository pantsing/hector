package internal

import (
	"github.com/urfave/cli"
	"github.com/pantsing/hector/internal/cmds/run"
)

var Commands []cli.Command = []cli.Command{
	run.Command(),
}
