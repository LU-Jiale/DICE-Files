package dbms.memory;


import dbms.memory.AbstractBufferManager.Policy;

public class BufferManager extends AbstractBufferManager {
	
	public BufferManager(Policy replPolicy, int numFrames,
			int pageSize, DiskSpaceManager spaceMan){
		super(replPolicy, numFrames, pageSize, spaceMan);
	}
	
	/* This method fetches a page with *disk* address 'pageAddr'. If the page
	 * is not in the buffer pool already, it must be retrieved from disk. The
	 * bookkeeping information must be updated accordingly and by taking into
	 * account the replacement policy in use. Remember to take care of dirty
	 * pages before replacing them.
	 */
	public int fetch(int pageAddr) throws FullBufferException{
		int fAddr = -1;
		int framesCount = 0;
		int frame = -1;
		byte[] page = new byte[pageSize];
		// Check if the content is in bufferPool
		for (int i = 0; i * pageSize < bufferPool.length; i++ ){
			if(bookkeeping.get(i).pageAddress == pageAddr){
				bookkeeping.get(i).pinCount += 1;
				if(replPolicy == Policy.LRU || replPolicy == Policy.MRU){
					frame = i;
				}
				fAddr = bookkeeping.get(i).frameAddress;
				break;
			}
		}
		
		// bufferPool empty
		if(fAddr == -1){
			for (int i = 0; i * pageSize < bufferPool.length; i++){			
				if (bookkeeping.get(i).isEmptyFrame()){
					bookkeeping.get(i).pageAddress = pageAddr;
					frame = i;
					byte[] pageContent = spaceManager.read(pageAddr, pageSize);
					for(int j = 0; j < pageSize; j++){
						bufferPool[bookkeeping.get(i).frameAddress + j] =  pageContent[j];
					}
					bookkeeping.get(i).pinCount += 1;
					fAddr = bookkeeping.get(i).frameAddress;
					break;
				}
				else
					framesCount++;
			}
		}
		
		// bufferPool no empty
		if(fAddr == -1){
			for (int i = 0; i * pageSize < bufferPool.length; i++){			
				if (bookkeeping.get(i).pinCount == 0){
					if (bookkeeping.get(i).dirty){
						
						for(int j = 0; j < pageSize; j++){
							page[j] = bufferPool[bookkeeping.get(i).frameAddress + j];
						}
						spaceManager.write(bookkeeping.get(i).pageAddress, page);
						bookkeeping.get(i).dirty = false;
					}
					bookkeeping.get(i).pageAddress = pageAddr;
					frame = i;
					byte[] pageContent = spaceManager.read(pageAddr, pageSize);
					for(int j = 0; j < pageSize; j++){
						bufferPool[bookkeeping.get(i).frameAddress + j] =  pageContent[j];
					}
					bookkeeping.get(i).pinCount += 1;
					fAddr = bookkeeping.get(i).frameAddress;
					break;
					
				}					
			}
			
		}
		
		// bufferPool full
		if(fAddr == -1){
			throw new FullBufferException();
		}
			
		// policy operation
		switch (this.replPolicy){
		case FIFO:
			if(frame != -1){
				FrameInfo element = bookkeeping.get(frame);
				if(framesCount != 4){
					bookkeeping.add(framesCount + 1, element);
				}
				else
					bookkeeping.add(element);
				
				bookkeeping.remove(element);
			}
			break;
		case LIFO:
			if(frame != -1){
				FrameInfo element = bookkeeping.get(frame);
				bookkeeping.remove(element);
				bookkeeping.add(0, element);
			}
			break;
		case LRU:
			if(framesCount==0){
				for (int i = 0; i * pageSize < bufferPool.length; i++){			
					if (!bookkeeping.get(i).isEmptyFrame()){
						framesCount++;
					}
				}
			}
			if(frame != -1 && framesCount!=0){
				FrameInfo element = bookkeeping.get(frame);
				if(framesCount != 4){
					bookkeeping.add(framesCount + 1, element);
				}
				else
					bookkeeping.add(element);
				
				bookkeeping.remove(frame);
			}
			break;
		case MRU:
			if(framesCount==0){
				for (int i = 0; i * pageSize < bufferPool.length; i++){			
					if (!bookkeeping.get(i).isEmptyFrame()){
						framesCount++;
					}
				}
			}
			if(frame != -1 && framesCount!=0){
				FrameInfo element = bookkeeping.get(frame);
				bookkeeping.remove(element);
				bookkeeping.add(0, element);
			}
			break;
		default:
			break;
		}
		return fAddr;
	}

	/* This method releases a page, indicating whether it has been modified or
	 * not. The bookkeeping information must be updated accordingly (but it is
	 * independent of the replacement policy in use).
	 */
	public void release(int frameAddr, boolean modified){
		int e = 0;
		for (int i = 0; i * pageSize < bufferPool.length; i++){	
			if(bookkeeping.get(i).frameAddress == frameAddr){
				e = i;
				break;
			}	
		}
		bookkeeping.get(e).dirty |= modified;
		if(bookkeeping.get(e).pinCount > 0)
			bookkeeping.get(e).pinCount -= 1;		
	}
}
