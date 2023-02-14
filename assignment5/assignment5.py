from dataclasses import dataclass
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("/Users/justincramer/Documents/Coding/CME241/RL-book/"))
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian
from rl.function_approx import FunctionApprox, LinearFunctionApprox
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State
from rl.policy import DeterministicPolicy
from rl.approximate_dynamic_programming import back_opt_vf_and_policy, \
    ValueFunctionApprox
from typing import Callable, Sequence, Tuple, Iterator

@dataclass(frozen=True)
class PriceAndShares:
    price: float
    shares: int
    x: float # Added field for X_t

@dataclass(frozen=True)
class OptimalOrderExecution:
    
    shares: int
    time_steps: int
    avg_exec_price_diff: Sequence[Callable[[PriceAndShares], float]]
    price_dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]]
    x_dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]] # Dynamics for X_t
    utility_func: Callable[[float], float]
    discount_factor: float
    func_approx: ValueFunctionApprox[PriceAndShares]
    initial_price_distribution: Distribution[float]
    initial_x_distribution: Distribution[float] # Distribution for X_0
    
    def get_mdp(self, t: int) -> MarkovDecisionProcess[PriceAndShares, int]:

        utility_f: Callable[[float], float] = self.utility_func
        price_diff: Sequence[Callable[[PriceAndShares], float]] = \
            self.avg_exec_price_diff
        dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]] = \
            self.price_dynamics
        x_dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]] = \
            self.x_dynamics
        steps: int = self.time_steps

        class OptimalExecutionMDP(MarkovDecisionProcess[PriceAndShares, int]):

            def step(
                self,
                p_r: NonTerminal[PriceAndShares],
                sell: int
            ) -> SampledDistribution[Tuple[State[PriceAndShares],
                                           float]]:

                def sr_sampler_func(
                    p_r=p_r,
                    sell=sell
                ) -> Tuple[State[PriceAndShares], float]:
                    p_s: PriceAndShares = PriceAndShares(
                        price=p_r.state.price,
                        shares=sell,
                        x=p_r.state.x
                    )
                    next_price: float = dynamics[t](p_s).sample()
                    next_rem: int = p_r.state.shares - sell
                    next_x: float = x_dynamics[t](p_s).sample()
                    next_state: PriceAndShares = PriceAndShares(
                        price=next_price,
                        shares=next_rem,
                        x=next_x
                    )
                    reward: float = utility_f(
                        sell * (p_r.state.price - price_diff[t](p_s))
                    )
                    return (NonTerminal(next_state), reward)

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=100
                )

            def actions(self, p_s: NonTerminal[PriceAndShares]) -> \
                    Iterator[int]:
                if t == steps - 1:
                    return iter([p_s.state.shares])
                else:
                    return iter(range(p_s.state.shares + 1))

        return OptimalExecutionMDP()

    def get_states_distribution(self, t: int) -> \
            SampledDistribution[NonTerminal[PriceAndShares]]:
        
        def states_sampler_func() -> NonTerminal[PriceAndShares]:
            price: float = self.initial_price_distribution.sample()
            rem: int = self.shares
            x: float = self.initial_x_distribution.sample()
            for i in range(t):
                sell: int = Choose(range(rem + 1)).sample()
                p_s = PriceAndShares(price=price, shares=rem, x=x)
                price = self.price_dynamics[i](p_s).sample()
                x = self.x_dynamics[i](p_s).sample() # Simulate X_t
                rem -= sell
            return NonTerminal(PriceAndShares(
                price=price,
                shares=rem,
                x=x
            ))

        return SampledDistribution(states_sampler_func)

    def backward_induction_vf_and_pi(
        self
    ) -> Iterator[Tuple[ValueFunctionApprox[PriceAndShares],
                        DeterministicPolicy[PriceAndShares, int]]]:
        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[PriceAndShares, int],
            ValueFunctionApprox[PriceAndShares],
            SampledDistribution[NonTerminal[PriceAndShares]]
        ]] = [(
            self.get_mdp(i),
            self.func_approx,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps)]
        num_state_samples: int = 100 # 10000
        error_tolerance: float = 1e-6
        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=self.discount_factor,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )
    
if __name__ == '__main__':
    init_price_mean: float = 100.0
    init_price_stdev: float = 10.0
    eta_stdev: float = 1
    z_mean: float = 0
    z_stdev: float = 1
    beta: float = 0.05
    theta: float = 0.01
    rho: float = 0.5
    # Not sure if I should use this for initial x distribution...
    init_x_mean: float = 0
    init_x_stdev: float = eta_stdev ** 2 / (1 - rho ** 2)
    num_shares: int = 100
    num_time_steps: int = 5
    alpha: float = 0.03
    # Temporary price impact
    price_diff = [lambda p_s: beta * p_s.shares * p_s.price + theta * p_s.x * p_s.price
                  for _ in range(num_time_steps)]
    # Distribution for next price given current price
    class ExponentialGaussian(SampledDistribution[float]):
        def __init__(self, price):
            z_distr = Gaussian(μ=z_mean, σ=z_stdev)
            super().__init__(lambda: price * np.e ** z_distr.sample())    

    dynamics = [lambda p_s: ExponentialGaussian(p_s.price) for _ in range(num_time_steps)]
    x_dynamics = [lambda p_s: Gaussian(μ=rho * p_s.x, σ=eta_stdev) for _ in range(num_time_steps)]
    # What should these be?
    ffs = [
        lambda p_s: p_s.state.price * p_s.state.shares,
        lambda p_s: float(p_s.state.shares * p_s.state.shares)
    ]
    fa: FunctionApprox = LinearFunctionApprox.create(feature_functions=ffs)

    init_price_distrib: Gaussian = Gaussian(
        μ=init_price_mean,
        σ=init_price_stdev
    )

    init_x_distrib: Gaussian = Gaussian(
        μ=init_x_mean,
        σ=init_x_stdev
    )

    ooe: OptimalOrderExecution = OptimalOrderExecution(
        shares=num_shares,
        time_steps=num_time_steps,
        avg_exec_price_diff=price_diff,
        price_dynamics=dynamics,
        x_dynamics=x_dynamics,
        utility_func=lambda x: x,
        discount_factor=1,
        func_approx=fa,
        initial_price_distribution=init_price_distrib,
        initial_x_distribution=init_x_distrib
    )
    it_vf: Iterator[Tuple[ValueFunctionApprox[PriceAndShares],
                          DeterministicPolicy[PriceAndShares, int]]] = \
        ooe.backward_induction_vf_and_pi()

    state: PriceAndShares = PriceAndShares(
        price=init_price_mean,
        shares=num_shares,
        x=init_x_mean
    )

    print("Backward Induction: VF And Policy")
    print("---------------------------------")
    print()
    for t, (vf, pol) in enumerate(it_vf):
        print(f"Time {t:d}")
        print()
        opt_sale: int = pol.action_for(state)
        val: float = vf(NonTerminal(state))
        print(f"Optimal Sales = {opt_sale:d}, Opt Val = {val:.3f}")
        print()
        print("Optimal Weights below:")
        print(vf.weights.weights)
        print()