#[cfg(test)]
mod tests {

    use crate::quick_grad::{grad_tape::GradTape, var::Var};

    fn round(x: f64, decimals: i32) -> f64 {
        let factor = 10.0_f64.powi(decimals as i32);
        (x * factor).round() / factor as f64
    }

    #[test]
    fn simple_ad() {
        let t = GradTape::new();
        let x = t.var(3.0);
        let y = t.var(5.0);

        let z = x * y;

        let grad = z.backward();

        assert_eq!(grad[&x], 5.0);
        assert_eq!(grad[&y], 3.0);
    }

    #[test]
    fn complex_ad() {
        let t = GradTape::new();

        let x = t.var(3.0);
        let y = t.var(5.0);
        let z = t.var(7.0);

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
        assert_eq!(round(grad[&x], 6), round(d_x, 6)); // Taking into account f64 fuckery
    }
    #[test]
    fn sigmoid_ad_with_constants() {
        let t = GradTape::new();

        let x = t.var(3.0);

        let sigmoid = |x: Var| 1.0 / (1.0 + (-x).exp());
        let y = sigmoid(x);
        let grad = y.backward();

        let d = {
            let x: f64 = 3.0;
            let sig = 1.0 / (1.0 + (-x).exp());

            sig * (1.0 - sig)
        };

        assert_eq!(round(grad[&x], 5), round(d, 5));
    }

    #[test]
    fn clear_tape() {
        let t = GradTape::new();

        let mut x = t.var(3.0);

        let sigmoid = |x: Var| 1.0 / (1.0 + (-x).exp());
        let y = sigmoid(x);
        let grad = y.backward();

        let d = {
            let x: f64 = 3.0;
            let sig = 1.0 / (1.0 + (-x).exp());

            sig * (1.0 - sig)
        };

        assert_eq!(round(grad[&x], 5), round(d, 5));

        t.clear(vec![&mut x]);

        // What if we don't have this?
        // let x = t.var(3.0);
        let sigmoid = |x: Var| 1.0 / (1.0 + (-x).exp());
        let y = sigmoid(x);
        let grad = y.backward();

        let d = {
            let x: f64 = 3.0;
            let sig = 1.0 / (1.0 + (-x).exp());

            sig * (1.0 - sig)
        };

        assert_eq!(round(grad[&x], 5), round(d, 5));
    }
    #[test]
    fn clear_tape_with_complex_ad() {
        let t = GradTape::new();

        let mut x = t.var(3.0);
        let mut y = t.var(5.0);
        let mut z = t.var(7.0);

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
        assert_eq!(round(grad[&x], 6), round(d_x, 6)); // Taking into account f64 fuckery

        t.clear(vec![&mut x, &mut y, &mut z]);

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
        assert_eq!(round(grad[&x], 6), round(d_x, 6)); // Taking into account f64 fuckery
    }
}
