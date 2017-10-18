package run

import (
	"github.com/pantsing/hector/internal/algorithms"
	"github.com/urfave/cli"
)

func Command() cli.Command {
	return cli.Command{
		Name:  "run",
		Usage: "Run one of  train,test or predict actions",
		Flags: []cli.Flag{
			cli.IntFlag{
				Name:  "verbose",
				Usage: "0: none; 1: verbose output",
			},
			cli.BoolFlag{
				Name:  "prof",
				Usage: "runtime profile",
			},
		},
		Subcommands: algorithms.Commands(),
	}
}
