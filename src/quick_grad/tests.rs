use super::grad_tape;
use super::node;

#[cfg(test)]
mod tests {
    use crate::quick_grad::grad_tape::GradTape;

    use super::*;

    #[test]
    fn simple_ad() {
        let tape = GradTape::new();
        let x = tape.var(3.0);
        let y = tape.var(5.0);

        let z = x * y;

        let grad = z.backward();

        assert_eq!(grad[x], 5.0);
        assert_eq!(grad[y], 3.0);
    }

    #[test]
    fn complex_ad() {
        let tape = GradTape::new();

        let x = tape.var(3.0);
        let y = tape.var(5.0);
        let z = tape.var(7.0);

        let r = ((x * y).cos() * (y * z + x).sin()) / x;
        let grad = r.backward();

        let d_x = {
            let x = x.value();
            let y = y.value();
            let z = z.value();

            -(y * (y * x).sin() * (x + y * z).sin() / x)
                - (((y * x).cos() * (x + y * z).sin()) / (x * x))
                + (((y * x).cos() * (x + y * z).cos()) / x)
        };
        assert_eq!(grad[x], d_x - 0.0000000000000001); // Taking into account f64 fuckery
    }
}
