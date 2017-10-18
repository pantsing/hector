package cluster

import (
	"github.com/pantsing/hector/internal/algorithms/internal"
	"github.com/pantsing/hector/internal/core"
	"github.com/urfave/cli"
)

type Cluster interface {
	internal.Algorithm
	Cluster(dataset core.DataSet)
}

var clusterIndex map[string]Cluster = map[string]Cluster{}

func Commands() []cli.Command {
	cmds := make([]cli.Command, 0, 1 << 5)
	for _, alog := range clusterIndex {
		cmd := alog.Command()
		if _, ok := internal.AlogCmdsChecker[cmd.Name]; ok {
			continue
		}
		internal.AlogCmdsChecker[cmd.Name] = struct{}{}
	}
	return cmds
}
