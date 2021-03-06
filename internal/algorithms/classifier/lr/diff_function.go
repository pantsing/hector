package lr

import (
	"github.com/pantsing/hector/internal/core"
)

// Description: function for minimizer such as LBFGS and OWLQN
type DiffFunction interface {
	Value(pos *core.Vector) float64
	Gradient(pos *core.Vector) *core.Vector
}
