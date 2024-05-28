import React from 'react'
import '../styles/Caption.css'

const CaptionGen = ({preds}) => {
  return (
    <div className='caption-wrapper'>
        <p className='instructions'>Suggested Impression:</p>
        {preds.map((pred, index) => {
            return <div key={index} className='diagnose-wrapper'>
                <p>{pred}</p>
            </div>
            })
        }
    </div>
  )
}

export default CaptionGen