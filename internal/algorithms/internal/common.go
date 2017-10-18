package internal

import (
	"github.com/urfave/cli"
)

type Algorithm interface {
	//Set parameters
	Init(ctx *cli.Context)
	Command() cli.Command
	Clear()
}

var AlogCmdsChecker map[string]struct{} = make(map[string]struct{})
