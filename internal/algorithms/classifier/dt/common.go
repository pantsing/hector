package dt

import "github.com/urfave/cli"

var commands []cli.Command = make([]cli.Command, 0, 1<<3)

func Commands() []cli.Command {
	return commands
}
