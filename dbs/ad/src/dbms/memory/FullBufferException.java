package dbms.memory;

public class FullBufferException extends Exception {

	private static final long serialVersionUID = -2216468143706905175L;

	public FullBufferException() {
		super();
	}
	
	public FullBufferException(String message) {
		super(message);
	}
	
	public FullBufferException(String message, Throwable cause) {
		super(message, cause);
	}
	
	public FullBufferException(Throwable cause) {
		super(cause);
	}  
}
