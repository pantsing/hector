package main

import (
	"github.com/pantsing/hector/internal"
	"github.com/urfave/cli"
	"os"
	"sort"
)

func main() {
	app := cli.NewApp()
	app.Name = "Hector"
	app.Description = "Machine Learning Algorithmes"
	app.Version = "1.0.0"
	app.Commands = internal.Commands
	sort.Sort(cli.CommandsByName(app.Commands))
	app.Run(os.Args)
}
