use super::grad_tape;
use super::node;

#[cfg(test)]
mod tests {
    use crate::quick_grad::grad_tape::GradTape;

    use super::*;

    #[test]
    fn test_node() {
        let mut tape = GradTape::new();
        let x = tape.var(3.0);
        let y = tape.var(5.0);

        let z = x * y;

        let grad = z.backward();
    }
}
