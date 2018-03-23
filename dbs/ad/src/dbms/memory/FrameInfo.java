package dbms.memory;

/* This class represent a row in the buffer manager's bookkeeping table.
 */
public final class FrameInfo {

	/* Do not rely on the specific value "-1" shown here.
	 * To check whether a frame is empty use the 'isEmptyFrame()' rather than
	 * checking whether pageAddress == -1
	 */
	private static final int EMPTY = -1;
	
	int frameAddress; // should always be non-negative
	int pageAddress = EMPTY;
	int pinCount = 0; // should always be non-negative
	boolean dirty = false;
	
	FrameInfo(int addr) {
		frameAddress = addr;
	}

	boolean isEmptyFrame() {
		return pageAddress == EMPTY;
	}
	
	@Override
	public String toString() {
		return String.format("%10d | %10d | %10d | %s", frameAddress, pageAddress, pinCount, dirty);
	}
}
