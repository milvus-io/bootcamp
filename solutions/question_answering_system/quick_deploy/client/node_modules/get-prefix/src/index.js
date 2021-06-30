export default function getPrefix(prop) {
  if (typeof document === 'undefined') return prop
  
  const styles = document.createElement('p').style
  const vendors = ['ms', 'O', 'Moz', 'Webkit']

  if (styles[prop] === '') return prop;

  prop = prop.charAt(0).toUpperCase() + prop.slice(1)

  for (let i = vendors.length; i--;) {
    if (styles[vendors[i] + prop] === '') {
      return (vendors[i] + prop)
    }
  }
}