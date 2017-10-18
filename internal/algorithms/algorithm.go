package algorithms

import (
	"github.com/pantsing/hector/internal/algorithms/classifier"
	"github.com/pantsing/hector/internal/algorithms/cluster"
	"github.com/pantsing/hector/internal/algorithms/regressor"
	"github.com/urfave/cli"
	"sort"
)

func Commands() []cli.Command {
	cmds := make([]cli.Command, 0, 1<<4)
	cmds = append(cmds, classifier.Commands()...)
	cmds = append(cmds, cluster.Commands()...)
	cmds = append(cmds, regressor.Commands()...)
	sort.Sort(cli.CommandsByName(cmds))
	return cmds
}
