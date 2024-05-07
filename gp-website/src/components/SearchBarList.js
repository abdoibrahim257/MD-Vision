import React from 'react'
import '../styles/searchResultList.css'

const SearchBarList = ({results}) => {
  return (
    <div className='result-list'>
        {results.map((result, id) => {
            return (
                <div className='search-result' key={id}>
                    {result}
                </div>
            );
        })}
    </div>
  )
}

export default SearchBarList