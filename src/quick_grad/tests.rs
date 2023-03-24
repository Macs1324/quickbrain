use super::node;
use super::operation;
use super::grad_tape;

#[cfg(test)]]
mod tests {
    use super::*;

    #[test]
    fn test_node() {
        let x = Node::new(1.0);
        let y = Node::new(2.0);
        let z = x + y;
        assert_eq!(z.value(), 3.0);
    }
}
