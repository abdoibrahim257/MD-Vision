import React , { useEffect }from 'react'
import "../styles/Loading.css"

const LoadingComponent = () => {

    useEffect(() => {
        // Add overflow: hidden to the body when the component is mounted
        document.body.style.overflow = 'hidden';
    
        // Remove overflow: hidden from the body when the component is unmounted
        return () => {
          document.body.style.overflow = 'unset';
        };
      }, []);

  return (
    <div class="loading"> 
        <svg width="16px" height="12px">
            <polyline id="back" points="1 6 4 6 6 11 10 1 12 6 15 6"></polyline>
            <polyline id="front" points="1 6 4 6 6 11 10 1 12 6 15 6"></polyline>
        </svg>
    </div>  
  )
}

export default LoadingComponent