import React from 'react'
import '../styles/searchResultList.css'
import { NavLink } from 'react-router-dom';

const SearchBarList = ({results}) => {
  return (
    <div className='result-list'>
        {results.map((result, id) => {
            return (
                <NavLink to={`/maven/${result}`} className='search-result' key={id}>
                      {result}
                </NavLink>
            );
        })}
    </div>
  )
}

export default SearchBarList